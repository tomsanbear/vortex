// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::GetResultPayload;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::DefaultHostAllocator;
use vortex_array::memory::HostAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::CoalesceConfig;
use crate::VortexReadAt;
use crate::runtime::Handle;
#[cfg(not(target_arch = "wasm32"))]
use crate::std_file::read_exact_at;

/// Default number of concurrent requests to allow.
pub const DEFAULT_CONCURRENCY: usize = 192;

/// An object store backed I/O source.
pub struct ObjectStoreReadAt {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    uri: Arc<str>,
    handle: Handle,
    allocator: HostAllocatorRef,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
}

impl ObjectStoreReadAt {
    /// Create a new object store source.
    pub fn new(store: Arc<dyn ObjectStore>, path: ObjectPath, handle: Handle) -> Self {
        Self::new_with_allocator(store, path, handle, Arc::new(DefaultHostAllocator))
    }

    /// Create a new object store source with a custom writable buffer allocator.
    pub fn new_with_allocator(
        store: Arc<dyn ObjectStore>,
        path: ObjectPath,
        handle: Handle,
        allocator: HostAllocatorRef,
    ) -> Self {
        let uri = Arc::from(path.to_string());
        Self {
            store,
            path,
            uri,
            handle,
            allocator,
            concurrency: DEFAULT_CONCURRENCY,
            coalesce_config: Some(CoalesceConfig::object_storage()),
        }
    }

    /// Set the concurrency for this source.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Set the coalesce config for this source.
    pub fn with_coalesce_config(mut self, config: CoalesceConfig) -> Self {
        self.coalesce_config = Some(config);
        self
    }
}

impl VortexReadAt for ObjectStoreReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.coalesce_config
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        async move {
            store
                .head(&path)
                .await
                .map(|h| h.size)
                .map_err(VortexError::from)
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let handle = self.handle.clone();
        let allocator = Arc::clone(&self.allocator);
        let range = offset..(offset + length as u64);

        // Requires to deal with borrowed lifetimes
        let io_handle = handle.clone();

        handle
                .spawn_io(async move {
                    let response = store
                        .get_opts(
                            &path,
                            GetOptions {
                                range: Some(GetRange::Bounded(range.clone())),
                                ..Default::default()
                            },
                        )
                        .await?;

                    let buffer = match response.payload {
                        #[cfg(not(target_arch = "wasm32"))]
                        GetResultPayload::File(file, _) => {
                            let mut buffer = allocator.allocate(length, alignment)?;
                            io_handle
                                .spawn_blocking(move || {
                                    read_exact_at(&file, buffer.as_mut_slice(), range.start)?;
                                    Ok::<_, io::Error>(buffer)
                                })
                                .await
                                .map_err(io::Error::other)?
                        }
                        #[cfg(target_arch = "wasm32")]
                        GetResultPayload::File(..) => {
                            unreachable!("File payload not supported on wasm32")
                        }
                        GetResultPayload::Stream(mut byte_stream) => {
                            let first = byte_stream.next().await.transpose()?;

                            // A single chunk covering the entire response (typical for in-memory
                            // and caching stores) can be adopted zero-copy once the stream is
                            // confirmed exhausted.
                            if let Some(bytes) = first.as_ref().filter(|bytes| bytes.len() == length)
                            {
                                while let Some(extra) = byte_stream.next().await {
                                    let extra = extra?;
                                    vortex_ensure!(
                                        extra.is_empty(),
                                        "Object store stream returned too many bytes: {} > expected {} (range: {:?})",
                                        length + extra.len(),
                                        length,
                                        range
                                    );
                                }
                                return match allocator.try_adopt(bytes, alignment) {
                                    Some(adopted) => {
                                        tracing::trace!(
                                            length,
                                            "adopted object store stream bytes zero-copy"
                                        );
                                        Ok(BufferHandle::new_host(adopted))
                                    }
                                    None => {
                                        tracing::trace!(
                                            length,
                                            "copied object store stream bytes: allocator declined adoption"
                                        );
                                        let mut buffer = allocator.allocate(length, alignment)?;
                                        buffer.as_mut_slice().copy_from_slice(bytes);
                                        Ok(BufferHandle::new_host(buffer.freeze()))
                                    }
                                };
                            }

                            let mut byte_stream =
                                stream::iter(first.map(Ok)).chain(byte_stream);
                            let mut buffer = allocator.allocate(length, alignment)?;
                            let mut written = 0usize;
                            while let Some(bytes) = byte_stream.next().await {
                                let bytes = bytes?;
                                let end = written + bytes.len();
                                vortex_ensure!(
                                    end <= length,
                                    "Object store stream returned too many bytes: {} > expected {} (range: {:?})",
                                    end,
                                    length,
                                    range
                                );
                                buffer.as_mut_slice()[written..end].copy_from_slice(&bytes);
                                written = end;
                            }

                            vortex_ensure!(
                                written == length,
                                "Object store stream returned {} bytes but expected {} bytes (range: {:?})",
                                written,
                                length,
                                range
                            );

                            tracing::trace!(
                                length,
                                "copied object store stream bytes: multi-chunk response"
                            );
                            buffer
                        }
                    };

                    Ok(BufferHandle::new_host(buffer.freeze()))
                })
        .boxed()
    }
}

#[cfg(test)]
mod tests {

    use std::ops::Range;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::stream::BoxStream;
    use object_store::CopyOptions;
    use object_store::GetResult;
    use object_store::ListResult;
    use object_store::MultipartUpload;
    use object_store::ObjectMeta;
    use object_store::PutMultipartOptions;
    use object_store::PutOptions;
    use object_store::PutPayload;
    use object_store::PutResult;
    use object_store::memory::InMemory;
    use rstest::rstest;
    use vortex_array::memory::AdoptingHostAllocator;
    use vortex_array::memory::HostAllocator;
    use vortex_array::memory::WritableHostBuffer;

    use super::*;
    use crate::runtime::AbortHandle;
    use crate::runtime::AbortHandleRef;
    use crate::runtime::Executor;

    const TEST_DATA: &[u8] = b"object store test data";

    #[derive(Default)]
    struct CountingExecutor {
        spawn_count: AtomicUsize,
        spawn_io_count: AtomicUsize,
    }

    impl Executor for CountingExecutor {
        fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
            self.spawn_count.fetch_add(1, Ordering::SeqCst);
            TokioAbortHandle::new_handle(tokio::spawn(fut).abort_handle())
        }

        fn spawn_io(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
            self.spawn_io_count.fetch_add(1, Ordering::SeqCst);
            TokioAbortHandle::new_handle(tokio::spawn(fut).abort_handle())
        }

        fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
            TokioAbortHandle::new_handle(tokio::spawn(async move { task() }).abort_handle())
        }

        fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
            TokioAbortHandle::new_handle(tokio::task::spawn_blocking(task).abort_handle())
        }
    }

    struct TokioAbortHandle(tokio::task::AbortHandle);

    impl TokioAbortHandle {
        fn new_handle(handle: tokio::task::AbortHandle) -> AbortHandleRef {
            Box::new(Self(handle))
        }
    }

    impl AbortHandle for TokioAbortHandle {
        fn abort(self: Box<Self>) {
            self.0.abort();
        }
    }

    #[tokio::test]
    async fn read_at_uses_spawn_io() -> anyhow::Result<()> {
        let executor = Arc::new(CountingExecutor::default());
        let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&runtime));

        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = ObjectPath::from("test.bin");
        store.put(&path, PutPayload::from_static(TEST_DATA)).await?;

        let reader = ObjectStoreReadAt::new(store, path, handle);
        let buffer = reader.read_at(7, 5, Alignment::new(1)).await?;

        assert_eq!(buffer.to_host().await.as_slice(), b"store");
        assert_eq!(executor.spawn_io_count.load(Ordering::SeqCst), 1);
        assert_eq!(executor.spawn_count.load(Ordering::SeqCst), 0);

        Ok(())
    }

    /// Serves a fixed sequence of stream chunks for any `get_opts` request.
    #[derive(Debug)]
    struct ChunkedStore {
        chunks: Vec<Bytes>,
    }

    impl std::fmt::Display for ChunkedStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "ChunkedStore")
        }
    }

    #[async_trait]
    impl ObjectStore for ChunkedStore {
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
            let size: u64 = self.chunks.iter().map(|chunk| chunk.len() as u64).sum();
            let range = match options.range {
                Some(GetRange::Bounded(range)) => range,
                _ => 0..size,
            };
            Ok(GetResult {
                payload: GetResultPayload::Stream(
                    stream::iter(self.chunks.clone().into_iter().map(Ok)).boxed(),
                ),
                meta: ObjectMeta {
                    location: location.clone(),
                    last_modified: Default::default(),
                    size,
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

    /// Delegates allocation to [`DefaultHostAllocator`] but never consents to adoption.
    #[derive(Debug, Default)]
    struct AdoptionRefusingAllocator {
        allocations: AtomicUsize,
    }

    impl HostAllocator for AdoptionRefusingAllocator {
        fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<WritableHostBuffer> {
            self.allocations.fetch_add(1, Ordering::SeqCst);
            DefaultHostAllocator.allocate(len, alignment)
        }
    }

    fn aligned_chunk(len: usize, alignment: Alignment) -> Bytes {
        let mut writable = DefaultHostAllocator.allocate(len, alignment).unwrap();
        for (idx, byte) in writable.as_mut_slice().iter_mut().enumerate() {
            *byte = u8::try_from(idx % 256).unwrap();
        }
        writable.freeze().into_inner()
    }

    fn chunked_reader(
        chunks: Vec<Bytes>,
        allocator: HostAllocatorRef,
    ) -> (Arc<dyn Executor>, ObjectStoreReadAt) {
        let executor: Arc<dyn Executor> = Arc::new(CountingExecutor::default());
        let handle = Handle::new(Arc::downgrade(&executor));
        let reader = ObjectStoreReadAt::new_with_allocator(
            Arc::new(ChunkedStore { chunks }),
            ObjectPath::from("test.bin"),
            handle,
            allocator,
        );
        (executor, reader)
    }

    #[tokio::test]
    async fn read_at_adopts_aligned_single_chunk() -> anyhow::Result<()> {
        let alignment = Alignment::new(64);
        let chunk = aligned_chunk(64, alignment);
        let (_executor, reader) =
            chunked_reader(vec![chunk.clone()], Arc::new(AdoptingHostAllocator));

        let buffer = reader.read_at(0, 64, alignment).await?.to_host().await;

        assert_eq!(buffer.as_slice(), chunk.as_ref());
        assert_eq!(buffer.as_ptr(), chunk.as_ptr());
        Ok(())
    }

    #[tokio::test]
    async fn read_at_copies_misaligned_single_chunk() -> anyhow::Result<()> {
        let alignment = Alignment::new(64);
        let chunk = aligned_chunk(65, alignment).slice(1..);
        let (_executor, reader) =
            chunked_reader(vec![chunk.clone()], Arc::new(AdoptingHostAllocator));

        let buffer = reader.read_at(0, 64, alignment).await?.to_host().await;

        assert_eq!(buffer.as_slice(), chunk.as_ref());
        assert!(buffer.is_aligned(alignment));
        assert_ne!(buffer.as_ptr(), chunk.as_ptr());
        Ok(())
    }

    #[tokio::test]
    async fn read_at_default_allocator_never_adopts() -> anyhow::Result<()> {
        let chunk = aligned_chunk(64, Alignment::new(64));
        let (_executor, reader) =
            chunked_reader(vec![chunk.clone()], Arc::new(DefaultHostAllocator));

        let buffer = reader
            .read_at(0, 64, Alignment::none())
            .await?
            .to_host()
            .await;

        assert_eq!(buffer.as_slice(), chunk.as_ref());
        assert_ne!(buffer.as_ptr(), chunk.as_ptr());
        Ok(())
    }

    #[tokio::test]
    async fn read_at_never_adopts_without_allocator_consent() -> anyhow::Result<()> {
        let allocator = Arc::new(AdoptionRefusingAllocator::default());
        let chunk = aligned_chunk(64, Alignment::new(64));
        let (_executor, reader) = chunked_reader(vec![chunk.clone()], Arc::clone(&allocator) as _);

        let buffer = reader
            .read_at(0, 64, Alignment::none())
            .await?
            .to_host()
            .await;

        assert_eq!(buffer.as_slice(), chunk.as_ref());
        assert_ne!(buffer.as_ptr(), chunk.as_ptr());
        assert_eq!(allocator.allocations.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn read_at_assembles_multi_chunk_stream() -> anyhow::Result<()> {
        let chunk = aligned_chunk(64, Alignment::new(64));
        let chunks = vec![chunk.slice(0..10), chunk.slice(10..64)];
        let (_executor, reader) = chunked_reader(chunks, Arc::new(AdoptingHostAllocator));

        let buffer = reader
            .read_at(0, 64, Alignment::none())
            .await?
            .to_host()
            .await;

        assert_eq!(buffer.as_slice(), chunk.as_ref());
        assert_ne!(buffer.as_ptr(), chunk.as_ptr());
        Ok(())
    }

    #[tokio::test]
    async fn read_at_empty_range() -> anyhow::Result<()> {
        let (_executor, reader) = chunked_reader(vec![], Arc::new(AdoptingHostAllocator));

        let buffer = reader
            .read_at(0, 0, Alignment::none())
            .await?
            .to_host()
            .await;

        assert!(buffer.is_empty());
        Ok(())
    }

    #[rstest]
    #[case::excess_after_full_chunk(vec![0..32, 32..64], 32, "too many bytes")]
    #[case::excess_within_chunk(vec![0..64], 32, "too many bytes")]
    #[case::short_stream(vec![0..16], 32, "expected 32 bytes")]
    #[tokio::test]
    async fn read_at_validates_stream_length(
        #[case] splits: Vec<Range<usize>>,
        #[case] length: usize,
        #[case] expected: &str,
    ) {
        let chunk = aligned_chunk(64, Alignment::new(64));
        let chunks = splits.into_iter().map(|range| chunk.slice(range)).collect();
        let (_executor, reader) = chunked_reader(chunks, Arc::new(AdoptingHostAllocator));

        let err = reader
            .read_at(0, length, Alignment::none())
            .await
            .unwrap_err();
        assert!(err.to_string().contains(expected), "{err}");
    }
}
