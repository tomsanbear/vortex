// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;

use bytes::BytesMut;
use futures::TryStreamExt;
use futures::stream::FuturesUnordered;
use object_store::MultipartUpload;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::PutPayload;
use object_store::PutResult;
use object_store::path::Path;
use vortex_error::VortexResult;

use crate::IoBuf;
use crate::VortexWrite;

/// Adapter type to write data through a [`ObjectStore`] instance.
///
/// After writing, the caller must make sure to call `shutdown`, in order to ensure the data is actually persisted.
///
/// The multipart upload is started **lazily** — only the first time the buffer actually
/// has to split into parts. An object that never splits is persisted by a single
/// [`ObjectStoreExt::put`] in `shutdown`: one request instead of the three a multipart
/// upload costs (create + part + complete), which is what object stores bill for. The
/// trigger is "the first required split", not a fixed object size: `write_all` only
/// splits past `BUFFER_SIZE`, while `flush` splits past `CHUNK_SIZE`, so how much a
/// given object buffers before splitting depends on the caller's write/flush pattern.
///
/// Deferring also means an object that never splits has no upload to orphan: if encoding
/// fails part-way, there is nothing left half-open on the store to be reaped later.
pub struct ObjectStoreWrite {
    object_store: Arc<dyn ObjectStore>,
    location: Path,
    /// `None` until the buffer first has to split — see `ensure_upload`.
    upload: Option<Box<dyn MultipartUpload>>,
    buffer: BytesMut,
    put_result: Option<PutResult>,
}

const CHUNK_SIZE: usize = 16 * 1024 * 1024;
const BUFFER_SIZE: usize = 128 * 1024 * 1024;

impl ObjectStoreWrite {
    pub async fn new(object_store: Arc<dyn ObjectStore>, location: &Path) -> VortexResult<Self> {
        Ok(Self {
            object_store,
            location: location.clone(),
            upload: None,
            buffer: BytesMut::with_capacity(CHUNK_SIZE),
            put_result: None,
        })
    }

    pub fn put_result(&self) -> Option<&PutResult> {
        self.put_result.as_ref()
    }

    /// Start the multipart upload unless it is already running.
    ///
    /// Returns `()` rather than a `&mut` to the upload on purpose: a method that hands
    /// back a reference borrows *all* of `self` for that reference's lifetime, which makes
    /// the `self.buffer.split_to(..)` in the split loops fail to borrow-check. Callers take
    /// the reference themselves afterwards, so `self.upload` and `self.buffer` stay
    /// independently borrowable and the parts can still be uploaded concurrently.
    async fn ensure_upload(&mut self) -> io::Result<()> {
        if self.upload.is_none() {
            // Clone both so no borrow of `self` spans the assignment back into `self.upload`.
            let object_store = Arc::clone(&self.object_store);
            let location = self.location.clone();
            self.upload = Some(object_store.put_multipart(&location).await?);
        }
        Ok(())
    }

    /// Drain every whole `CHUNK_SIZE` part out of the buffer, starting the upload on
    /// first use.
    ///
    /// The remainder always stays buffered, so every part emitted here is exactly
    /// `CHUNK_SIZE` and only `shutdown`'s trailing part can be smaller — object stores
    /// require every part but the last to clear a minimum size.
    async fn drain_parts(&mut self) -> io::Result<()> {
        if self.buffer.len() <= CHUNK_SIZE {
            return Ok(());
        }
        self.ensure_upload().await?;

        // `ensure_upload` just guaranteed `Some`. Were it somehow `None`, the bytes stay
        // buffered and `shutdown` still persists them, so this cannot silently drop data.
        if let Some(upload) = self.upload.as_mut() {
            let parts = FuturesUnordered::new();
            while self.buffer.len() > CHUNK_SIZE {
                let payload = self.buffer.split_to(CHUNK_SIZE).freeze();
                parts.push(upload.put_part(PutPayload::from_bytes(payload)));
            }
            parts.try_collect::<Vec<_>>().await?;
        }
        Ok(())
    }
}

impl VortexWrite for ObjectStoreWrite {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        self.buffer.extend_from_slice(buffer.as_slice());

        // Only start splitting once the buffer is full: holding up to BUFFER_SIZE keeps
        // small and mid-sized objects on the single-`put` path.
        if self.buffer.len() > BUFFER_SIZE {
            self.drain_parts().await?;
        }

        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        self.drain_parts().await
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        // A second call must not re-`put` the (now empty) buffer: on the single-`put`
        // path that would overwrite the object just written with zero bytes.
        if self.put_result.is_some() {
            return Ok(());
        }

        self.flush().await?;

        let bytes = std::mem::take(&mut self.buffer).freeze();
        let put_result = if let Some(upload) = self.upload.as_mut() {
            if !bytes.is_empty() {
                upload.put_part(PutPayload::from_bytes(bytes)).await?;
            }
            upload.complete().await?
        } else {
            // Never split, so the whole object is still buffered: one request persists it.
            self.object_store
                .put(&self.location, PutPayload::from_bytes(bytes))
                .await?
        };

        self.put_result = Some(put_result);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Display;
    use std::fmt::Formatter;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use object_store::CopyOptions;
    use object_store::GetOptions;
    use object_store::GetResult;
    use object_store::ListResult;
    use object_store::ObjectMeta;
    use object_store::ObjectStore;
    use object_store::PutMultipartOptions;
    use object_store::PutOptions;
    use object_store::UploadPart;
    use object_store::local::LocalFileSystem;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use rstest::rstest;
    use tempfile::tempdir;

    use super::*;

    // Note: Concurrent writes test removed because &mut self in write_all already ensures
    // exclusive access. Multiple writers would need to be created with separate buffers,
    // which is not the intended use case.

    #[tokio::test]
    #[rstest]
    #[case(100)]
    #[case(8 * 1024 * 1024)]
    #[case(25 * 1024 * 1024)]
    #[case(26 * 1024 * 1024)]
    async fn test_object_store_writer_multiple_flushes(
        #[case] chunk_size: usize,
    ) -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let local_store =
            Arc::new(LocalFileSystem::new_with_prefix(temp_dir.path())?) as Arc<dyn ObjectStore>;
        let memory_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let location = Path::from("test.bin");

        for test_store in [memory_store, local_store] {
            let mut writer = ObjectStoreWrite::new(Arc::clone(&test_store), &location).await?;

            #[expect(clippy::cast_possible_truncation)]
            let data = (0..3)
                .map(|i| vec![i as u8; chunk_size])
                .collect::<Vec<_>>();

            // Write and flush multiple times
            for i in 0..3 {
                let data = data[i].clone();
                writer.write_all(data).await?;
                writer.flush().await?;
            }

            // Shutdown the writer to make sure data actually gets persisted.
            writer.shutdown().await?;

            // Verify all data was written
            let result = test_store.get(&location).await?;
            let bytes = result.bytes().await?;

            let expected_data = itertools::concat(data.into_iter());
            assert_eq!(bytes, expected_data);
        }

        Ok(())
    }

    /// Counts the write requests an [`ObjectStore`] actually receives, so the tests below
    /// can assert on request *shape* rather than only on the resulting bytes.
    #[derive(Debug)]
    struct CountingStore {
        inner: Arc<dyn ObjectStore>,
        counts: Arc<WriteCounts>,
    }

    #[derive(Debug, Default)]
    struct WriteCounts {
        puts: AtomicUsize,
        multiparts: AtomicUsize,
        parts: AtomicUsize,
    }

    impl WriteCounts {
        fn snapshot(&self) -> (usize, usize, usize) {
            (
                self.puts.load(Ordering::SeqCst),
                self.multiparts.load(Ordering::SeqCst),
                self.parts.load(Ordering::SeqCst),
            )
        }
    }

    impl Display for CountingStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "CountingStore({})", self.inner)
        }
    }

    #[async_trait]
    impl ObjectStore for CountingStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            self.counts.puts.fetch_add(1, Ordering::SeqCst);
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            self.counts.multiparts.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(CountingUpload {
                inner: self.inner.put_multipart_opts(location, opts).await?,
                counts: Arc::clone(&self.counts),
            }))
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            self.inner.get_opts(location, options).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    #[derive(Debug)]
    struct CountingUpload {
        inner: Box<dyn MultipartUpload>,
        counts: Arc<WriteCounts>,
    }

    #[async_trait]
    impl MultipartUpload for CountingUpload {
        fn put_part(&mut self, data: PutPayload) -> UploadPart {
            self.counts.parts.fetch_add(1, Ordering::SeqCst);
            self.inner.put_part(data)
        }

        async fn complete(&mut self) -> object_store::Result<PutResult> {
            self.inner.complete().await
        }

        async fn abort(&mut self) -> object_store::Result<()> {
            self.inner.abort().await
        }
    }

    fn counting_store() -> (Arc<dyn ObjectStore>, Arc<WriteCounts>) {
        let counts = Arc::new(WriteCounts::default());
        let store = Arc::new(CountingStore {
            inner: Arc::new(InMemory::new()),
            counts: Arc::clone(&counts),
        }) as Arc<dyn ObjectStore>;
        (store, counts)
    }

    /// An object that never has to split is persisted by ONE `put`, not the three
    /// requests (create + part + complete) a multipart upload bills for.
    #[tokio::test]
    async fn small_object_costs_one_put_and_no_multipart() -> anyhow::Result<()> {
        let (store, counts) = counting_store();
        let location = Path::from("small.bin");

        let mut writer = ObjectStoreWrite::new(Arc::clone(&store), &location).await?;
        writer.write_all(vec![7u8; 4096]).await?;
        writer.flush().await?;
        writer.shutdown().await?;

        assert_eq!(counts.snapshot(), (1, 0, 0));
        let bytes = store.get(&location).await?.bytes().await?;
        assert_eq!(bytes, vec![7u8; 4096]);
        Ok(())
    }

    /// Once the buffer must split, the writer starts a real multipart upload and every
    /// part but the trailing one is a full `CHUNK_SIZE` — object stores impose a minimum
    /// size on all non-final parts.
    #[tokio::test]
    async fn splitting_object_uses_multipart_with_full_parts() -> anyhow::Result<()> {
        let (store, counts) = counting_store();
        let location = Path::from("large.bin");
        let tail = 4096;

        let mut writer = ObjectStoreWrite::new(Arc::clone(&store), &location).await?;
        writer.write_all(vec![3u8; CHUNK_SIZE + tail]).await?;
        writer.flush().await?;
        writer.shutdown().await?;

        // One create, one full part at flush, one trailing part at shutdown, no plain put.
        assert_eq!(counts.snapshot(), (0, 1, 2));
        let bytes = store.get(&location).await?.bytes().await?;
        assert_eq!(bytes.len(), CHUNK_SIZE + tail);
        Ok(())
    }

    /// A second `shutdown` must not re-`put` the drained buffer over the object just
    /// written — on the single-`put` path that would truncate it to zero bytes.
    #[tokio::test]
    async fn second_shutdown_does_not_truncate() -> anyhow::Result<()> {
        let (store, counts) = counting_store();
        let location = Path::from("twice.bin");

        let mut writer = ObjectStoreWrite::new(Arc::clone(&store), &location).await?;
        writer.write_all(vec![9u8; 2048]).await?;
        writer.shutdown().await?;
        writer.shutdown().await?;

        assert_eq!(
            counts.snapshot(),
            (1, 0, 0),
            "the second shutdown must be a no-op"
        );
        let bytes = store.get(&location).await?.bytes().await?;
        assert_eq!(bytes, vec![9u8; 2048]);
        Ok(())
    }

    /// A writer that never received bytes still persists a zero-byte object, matching
    /// what completing a zero-part multipart upload produced.
    #[tokio::test]
    async fn empty_writer_persists_zero_byte_object() -> anyhow::Result<()> {
        let (store, counts) = counting_store();
        let location = Path::from("empty.bin");

        let mut writer = ObjectStoreWrite::new(Arc::clone(&store), &location).await?;
        writer.shutdown().await?;

        assert_eq!(counts.snapshot(), (1, 0, 0));
        let bytes = store.get(&location).await?.bytes().await?;
        assert!(bytes.is_empty());
        Ok(())
    }
}
