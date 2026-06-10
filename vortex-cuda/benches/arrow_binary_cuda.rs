// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for Arrow Device export of binary view arrays as Arrow Binary.

#![expect(clippy::cast_possible_truncation)]

#[allow(dead_code)]
mod bench_config;
mod timed_launch_strategy;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::PushKernelArg;
use futures::executor::block_on;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::varbinview::BinaryView;
use vortex::array::buffer::BufferHandle;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::session::VortexSession;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::CudaSessionExt;
use vortex_cuda::arrow::ArrowDeviceArray;
use vortex_cuda::arrow::DeviceArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::timed_launch_strategy::TimedLaunchStrategy;

const BINARY_BENCH_SIZES: &[(usize, &str)] = &[(10_000_000, "10M")];
const VALIDITY_REPACK_OFFSETS: &[(usize, &str)] = &[(1, "offset_1"), (7, "offset_7")];

async fn binary_on_device(
    views: Buffer<BinaryView>,
    buffers: Arc<[ByteBuffer]>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<ArrayRef> {
    binary_on_device_with_validity(views, buffers, Validity::NonNullable, ctx).await
}

async fn binary_on_device_with_validity(
    views: Buffer<BinaryView>,
    buffers: Arc<[ByteBuffer]>,
    validity: Validity,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<ArrayRef> {
    let views = ctx
        .ensure_on_device(BufferHandle::new_host(views.into_byte_buffer()))
        .await?;
    let mut device_buffers = Vec::with_capacity(buffers.len());
    for buffer in buffers.iter() {
        device_buffers.push(
            ctx.ensure_on_device(BufferHandle::new_host(buffer.clone()))
                .await?,
        );
    }

    Ok(VarBinViewArray::new_handle(
        views,
        device_buffers.into(),
        DType::Binary(validity.nullability()),
        validity,
    )
    .into_array())
}

async fn inline_binary(len: usize, ctx: &mut CudaExecutionCtx) -> VortexResult<ArrayRef> {
    binary_on_device(
        Buffer::from_iter((0..len).map(|idx| BinaryView::make_view(&idx.to_le_bytes(), 0, 0))),
        Arc::from([]),
        ctx,
    )
    .await
}

async fn out_of_line_binary(len: usize, ctx: &mut CudaExecutionCtx) -> VortexResult<ArrayRef> {
    let values = ByteBuffer::copy_from(vec![b'x'; len * 16]);
    let views = Buffer::from_iter((0..len).map(|idx| {
        let offset = idx * 16;
        BinaryView::make_view(&values.slice(offset..offset + 16), 0, offset as u32)
    }));

    binary_on_device(views, Arc::from([values]), ctx).await
}

async fn sliced_validity_binary(
    len: usize,
    bit_offset: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<ArrayRef> {
    let views =
        Buffer::from_iter((0..len).map(|idx| BinaryView::make_view(&idx.to_le_bytes(), 0, 0)));
    let validity = Validity::from_iter((0..len + bit_offset).map(|idx| idx % 3 != 0))
        .slice(bit_offset..bit_offset + len)?;

    binary_on_device_with_validity(views, Arc::from([]), validity, ctx).await
}

fn validity_bitmap(len: usize, bit_offset: usize) -> ByteBuffer {
    let total_bits = len + bit_offset;
    let mut bitmap = vec![0u8; total_bits.div_ceil(8)];
    for idx in 0..total_bits {
        if idx % 3 != 0 {
            bitmap[idx / 8] |= 1 << (idx % 8);
        }
    }
    ByteBuffer::copy_from(bitmap)
}

unsafe fn release_arrow_device_array(array: &mut ArrowDeviceArray) {
    unsafe {
        if let Some(release) = array.array.release {
            release(&raw mut array.array);
        }
    }
}

fn benchmark_arrow_binary_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for &(len, len_label) in BINARY_BENCH_SIZES {
        group.throughput(Throughput::Bytes(
            (len * (size_of::<BinaryView>() + 8)) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new("cuda/arrow_binary/inline", len_label),
            &len,
            |b, &len| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context")
                        .with_launch_strategy(Arc::new(timed));
                    let array = block_on(inline_binary(len, &mut cuda_ctx))
                        .vortex_expect("failed to create binary fixture");

                    for _ in 0..iters {
                        let mut exported =
                            block_on(array.clone().export_device_array(&mut cuda_ctx))
                                .vortex_expect("failed to export device array");
                        unsafe { release_arrow_device_array(&mut exported) };
                    }

                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );

        group.throughput(Throughput::Bytes(
            (len * (size_of::<BinaryView>() + 16 + 8)) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new("cuda/arrow_binary/out_of_line", len_label),
            &len,
            |b, &len| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context")
                        .with_launch_strategy(Arc::new(timed));
                    let array = block_on(out_of_line_binary(len, &mut cuda_ctx))
                        .vortex_expect("failed to create binary fixture");

                    for _ in 0..iters {
                        let mut exported =
                            block_on(array.clone().export_device_array(&mut cuda_ctx))
                                .vortex_expect("failed to export device array");
                        unsafe { release_arrow_device_array(&mut exported) };
                    }

                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );

        group.throughput(Throughput::Bytes(
            (len * (size_of::<BinaryView>() + 8) + len.div_ceil(8)) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new("cuda/arrow_binary/sliced_validity", len_label),
            &len,
            |b, &len| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context")
                        .with_launch_strategy(Arc::new(timed));
                    let array = block_on(sliced_validity_binary(len, 1, &mut cuda_ctx))
                        .vortex_expect("failed to create binary fixture");

                    for _ in 0..iters {
                        let mut exported =
                            block_on(array.clone().export_device_array(&mut cuda_ctx))
                                .vortex_expect("failed to export device array");
                        unsafe { release_arrow_device_array(&mut exported) };
                    }

                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );
    }

    for &(len, len_label) in BINARY_BENCH_SIZES {
        let bit_offset = 7;
        let bit_offset_label = "offset_7";
        group.throughput(Throughput::Bytes(
            (len * (size_of::<BinaryView>() + 8) + len.div_ceil(8)) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new(
                format!("cuda/arrow_binary/sliced_validity/{bit_offset_label}"),
                len_label,
            ),
            &(len, bit_offset),
            |b, &(len, bit_offset)| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context")
                        .with_launch_strategy(Arc::new(timed));
                    let array = block_on(sliced_validity_binary(len, bit_offset, &mut cuda_ctx))
                        .vortex_expect("failed to create binary fixture");

                    for _ in 0..iters {
                        let mut exported =
                            block_on(array.clone().export_device_array(&mut cuda_ctx))
                                .vortex_expect("failed to export device array");
                        unsafe { release_arrow_device_array(&mut exported) };
                    }

                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );
    }

    group.finish();
}

fn benchmark_arrow_binary_repack_validity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for &(len, len_label) in BINARY_BENCH_SIZES {
        let output_bytes = len.div_ceil(8);
        for &(bit_offset, bit_offset_label) in VALIDITY_REPACK_OFFSETS {
            group.throughput(Throughput::Bytes((output_bytes * 2) as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("cuda/arrow_binary/repack_validity/{bit_offset_label}"),
                    len_label,
                ),
                &(len, bit_offset),
                |b, &(len, bit_offset)| {
                    b.iter_custom(|iters| {
                        let timed = TimedLaunchStrategy::default();
                        let timer = timed.timer();

                        let vortex_session = VortexSession::empty();
                        let mut cuda_ctx = CudaSession::create_execution_ctx(&vortex_session)
                            .vortex_expect("failed to create execution context")
                            .with_launch_strategy(Arc::new(timed));
                        let kernel = vortex_session
                            .cuda_session()
                            .load_function_with_suffixes("arrow_binary", &["repack_validity"])
                            .vortex_expect("failed to load repack validity kernel");

                        let input = block_on(cuda_ctx.ensure_on_device(BufferHandle::new_host(
                            validity_bitmap(len, bit_offset),
                        )))
                        .vortex_expect("failed to copy validity bitmap to device");
                        let input_view = input
                            .cuda_view::<u8>()
                            .vortex_expect("failed to view validity bitmap");
                        let output = cuda_ctx
                            .device_alloc::<u8>(output_bytes.max(1))
                            .vortex_expect("failed to allocate repacked validity bitmap");
                        let len_u64 = len as u64;
                        let bit_offset_u64 = bit_offset as u64;
                        let output_bytes_u64 = output_bytes as u64;

                        for _ in 0..iters {
                            cuda_ctx
                                .launch_kernel(&kernel, output_bytes, |args| {
                                    args.arg(&input_view)
                                        .arg(&output)
                                        .arg(&len_u64)
                                        .arg(&bit_offset_u64)
                                        .arg(&output_bytes_u64);
                                })
                                .vortex_expect("failed to launch repack validity kernel");
                        }

                        Duration::from_nanos(timer.load(Ordering::Relaxed))
                    });
                },
            );
        }
    }

    group.finish();
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_arrow_binary_export, benchmark_arrow_binary_repack_validity
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
