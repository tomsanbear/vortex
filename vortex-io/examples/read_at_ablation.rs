// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Ablation harness for the zero-copy object store read path.
//!
//! Serves single-chunk stream responses from an in-memory buffer (the shape a caching
//! `ObjectStore` produces on a hit) and drives `read_at` in a loop under one of two allocators:
//!
//! - `--mode adopt`: `AdoptingHostAllocator` (opt-in), which adopts aligned single-chunk bytes
//!   zero-copy.
//! - `--mode copy`: `DefaultHostAllocator`, the default allocate-and-copy behavior.
//!
//! Run both modes under samply and compare the `memmove`/`memcpy` share:
//!
//! ```sh
//! cargo build --release --example read_at_ablation -p vortex-io --features object_store,tokio
//! samply record -o copy.json  ./target/release/examples/read_at_ablation --mode copy
//! samply record -o adopt.json ./target/release/examples/read_at_ablation --mode adopt
//! ```

use std::fmt::Debug;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use anyhow::bail;
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use object_store::CopyOptions;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::GetResult;
use object_store::GetResultPayload;
use object_store::ListResult;
use object_store::MultipartUpload;
use object_store::ObjectMeta;
use object_store::ObjectStore;
use object_store::PutMultipartOptions;
use object_store::PutOptions;
use object_store::PutPayload;
use object_store::PutResult;
use object_store::path::Path as ObjectPath;
use vortex_array::memory::AdoptingHostAllocator;
use vortex_array::memory::DefaultHostAllocator;
use vortex_array::memory::HostAllocator;
use vortex_array::memory::HostAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexExpect;
use vortex_io::VortexReadAt;
use vortex_io::object_store::ObjectStoreReadAt;
use vortex_io::runtime::tokio::TokioRuntime;

/// Serves any bounded range request as a single-chunk stream slice of one resident buffer,
/// mimicking a caching object store on a hit.
#[derive(Debug)]
struct ResidentStore {
    data: Bytes,
}

impl std::fmt::Display for ResidentStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ResidentStore")
    }
}

#[async_trait]
impl ObjectStore for ResidentStore {
    async fn put_opts(
        &self,
        _location: &ObjectPath,
        _payload: PutPayload,
        _opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        unimplemented!()
    }

    async fn put_multipart_opts(
        &self,
        _location: &ObjectPath,
        _opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        unimplemented!()
    }

    async fn get_opts(
        &self,
        location: &ObjectPath,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        let range = match options.range {
            Some(GetRange::Bounded(range)) => range,
            _ => 0..self.data.len() as u64,
        };
        let start = usize::try_from(range.start).vortex_expect("range start fits in usize");
        let end = usize::try_from(range.end).vortex_expect("range end fits in usize");
        let slice = self.data.slice(start..end);
        Ok(GetResult {
            payload: GetResultPayload::Stream(stream::iter([Ok(slice)]).boxed()),
            meta: ObjectMeta {
                location: location.clone(),
                last_modified: Default::default(),
                size: self.data.len() as u64,
                e_tag: None,
                version: None,
            },
            range,
            attributes: Default::default(),
        })
    }

    fn delete_stream(
        &self,
        _locations: BoxStream<'static, object_store::Result<ObjectPath>>,
    ) -> BoxStream<'static, object_store::Result<ObjectPath>> {
        unimplemented!()
    }

    fn list(
        &self,
        _prefix: Option<&ObjectPath>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        unimplemented!()
    }

    async fn list_with_delimiter(
        &self,
        _prefix: Option<&ObjectPath>,
    ) -> object_store::Result<ListResult> {
        unimplemented!()
    }

    async fn copy_opts(
        &self,
        _from: &ObjectPath,
        _to: &ObjectPath,
        _options: CopyOptions,
    ) -> object_store::Result<()> {
        unimplemented!()
    }
}

struct Config {
    mode: String,
    total_bytes: usize,
    read_size: usize,
    iters: usize,
    concurrency: usize,
    touch: bool,
}

fn parse_args() -> anyhow::Result<Config> {
    let mut config = Config {
        mode: String::new(),
        total_bytes: 256 << 20,
        read_size: 16 << 10,
        iters: 500_000,
        concurrency: 1,
        touch: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" => config.mode = args.next().context("--mode requires a value")?,
            "--total-mb" => {
                config.total_bytes = args
                    .next()
                    .context("--total-mb requires a value")?
                    .parse::<usize>()?
                    << 20
            }
            "--read-size" => {
                config.read_size = args
                    .next()
                    .context("--read-size requires a value")?
                    .parse()?
            }
            "--iters" => config.iters = args.next().context("--iters requires a value")?.parse()?,
            "--concurrency" => {
                config.concurrency = args
                    .next()
                    .context("--concurrency requires a value")?
                    .parse()?
            }
            "--touch" => config.touch = true,
            other => bail!("unknown argument: {other}"),
        }
    }
    if config.mode != "adopt" && config.mode != "copy" {
        bail!("--mode must be 'adopt' or 'copy'");
    }
    Ok(config)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = parse_args()?;

    let alignment = Alignment::new(8);
    let allocator: HostAllocatorRef = match config.mode.as_str() {
        "adopt" => Arc::new(AdoptingHostAllocator),
        _ => Arc::new(DefaultHostAllocator),
    };

    // Page-aligned resident buffer so every 8-byte-aligned offset yields adoptable slices.
    let mut writable = DefaultHostAllocator.allocate(config.total_bytes, Alignment::new(4096))?;
    for (idx, byte) in writable.as_mut_slice().iter_mut().enumerate() {
        *byte = u8::try_from(idx % 251)?;
    }
    let data = writable.freeze().into_inner();

    let reader = ObjectStoreReadAt::new_with_allocator(
        Arc::new(ResidentStore { data: data.clone() }),
        ObjectPath::from("resident.bin"),
        TokioRuntime::current(),
        allocator,
    );

    // Probe read: confirm whether this mode actually adopts (pointer identity with the store).
    let probe = reader
        .read_at(0, config.read_size, alignment)
        .await?
        .to_host()
        .await;
    let zero_copy = std::ptr::eq(probe.as_ptr(), data.as_ptr());
    drop(probe);

    // Deterministic pseudo-random aligned offsets within the resident buffer.
    let span = (config.total_bytes - config.read_size) as u64;
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut offsets = Vec::with_capacity(config.iters);
    for _ in 0..config.iters {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        offsets.push((state % span) & !7);
    }

    let mut bytes_read = 0usize;
    let mut checksum = 0u64;
    let start = Instant::now();
    let mut reads = stream::iter(offsets)
        .map(|offset| {
            let read = reader.read_at(offset, config.read_size, alignment);
            async move { Ok::<_, vortex_error::VortexError>(read.await?.to_host().await) }
        })
        .buffer_unordered(config.concurrency);
    while let Some(buffer) = reads.next().await {
        let buffer = buffer?;
        bytes_read += buffer.len();
        if config.touch {
            checksum = checksum.wrapping_add(
                buffer
                    .as_slice()
                    .iter()
                    .step_by(64)
                    .map(|byte| u64::from(*byte))
                    .sum::<u64>(),
            );
        }
        std::hint::black_box(&buffer);
    }
    let elapsed = start.elapsed();

    let gib = bytes_read as f64 / (1u64 << 30) as f64;
    println!(
        "mode={} zero_copy_verified={} reads={} read_size={} concurrency={} touch={} total_mb={} elapsed={:.3}s throughput={:.2}GiB/s reads_per_sec={:.0} checksum={}",
        config.mode,
        zero_copy,
        config.iters,
        config.read_size,
        config.concurrency,
        config.touch,
        config.total_bytes >> 20,
        elapsed.as_secs_f64(),
        gib / elapsed.as_secs_f64(),
        config.iters as f64 / elapsed.as_secs_f64(),
        checksum,
    );
    Ok(())
}
