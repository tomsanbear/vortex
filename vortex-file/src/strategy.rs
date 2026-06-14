// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the default layout strategy for a Vortex file.

use std::sync::Arc;
use std::sync::LazyLock;

use vortex_alp::ALP;
use vortex_alp::ALPRD;
use vortex_array::ArrayId;
use vortex_array::VTable;
use vortex_array::arrays::Bool;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Decimal;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Extension;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::List;
use vortex_array::arrays::ListView;
use vortex_array::arrays::Masked;
use vortex_array::arrays::Null;
use vortex_array::arrays::Patched;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::Struct;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::Variant;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::dtype::FieldPath;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::integer::IntDictScheme;
use vortex_bytebool::ByteBool;
use vortex_datetime_parts::DateTimeParts;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::Delta;
use vortex_fastlanes::FoR;
use vortex_fastlanes::RLE;
use vortex_fsst::FSST;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::buffered::BufferedStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::collect::CollectStrategy;
use vortex_layout::layouts::compressed::CompressingStrategy;
use vortex_layout::layouts::compressed::CompressorPlugin;
use vortex_layout::layouts::dict::writer::DictStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::repartition::RepartitionStrategy;
use vortex_layout::layouts::repartition::RepartitionWriterOptions;
use vortex_layout::layouts::table::TableStrategy;
use vortex_layout::layouts::zoned::writer::ZonedLayoutOptions;
use vortex_layout::layouts::zoned::writer::ZonedStrategy;
#[cfg(feature = "unstable_encodings")]
use vortex_onpair::OnPair;
use vortex_pco::Pco;
use vortex_runend::RunEnd;
use vortex_sequence::Sequence;
use vortex_sparse::Sparse;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;
use vortex_zigzag::ZigZag;
#[cfg(feature = "zstd")]
use vortex_zstd::Zstd;
#[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
use vortex_zstd::ZstdBuffers;

const ONE_MEG: u64 = 1 << 20;

/// Static registry of all allowed array encodings for file writing.
///
/// This includes all canonical encodings from vortex-array plus all compressed
/// encodings from the various encoding crates.
pub static ALLOWED_ENCODINGS: LazyLock<HashSet<ArrayId>> = LazyLock::new(|| {
    let mut allowed = HashSet::new();

    // Canonical encodings from vortex-array
    allowed.insert(Null.id());
    allowed.insert(Bool.id());
    allowed.insert(Primitive.id());
    allowed.insert(Decimal.id());
    allowed.insert(VarBin.id());
    allowed.insert(VarBinView.id());
    allowed.insert(List.id());
    allowed.insert(ListView.id());
    allowed.insert(FixedSizeList.id());
    allowed.insert(Struct.id());
    allowed.insert(Extension.id());
    allowed.insert(Chunked.id());
    allowed.insert(Constant.id());
    allowed.insert(Masked.id());
    allowed.insert(Dict.id());
    allowed.insert(Variant.id());

    // Compressed encodings from encoding crates
    allowed.insert(ALP.id());
    allowed.insert(ALPRD.id());
    allowed.insert(BitPacked.id());
    allowed.insert(ByteBool.id());
    allowed.insert(DateTimeParts.id());
    allowed.insert(DecimalByteParts.id());
    allowed.insert(Delta.id());
    allowed.insert(FoR.id());
    allowed.insert(FSST.id());
    #[cfg(feature = "unstable_encodings")]
    allowed.insert(OnPair.id());
    allowed.insert(Pco.id());
    allowed.insert(RLE.id());
    allowed.insert(RunEnd.id());
    allowed.insert(Sequence.id());
    allowed.insert(Sparse.id());
    allowed.insert(ZigZag.id());

    // Experimental encodings

    if use_experimental_patches() {
        allowed.insert(Patched.id());
    }

    #[cfg(feature = "zstd")]
    allowed.insert(Zstd.id());
    #[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
    allowed.insert(ZstdBuffers.id());

    allowed
});

/// How the compressor was configured on [`WriteStrategyBuilder`].
enum CompressorConfig {
    /// A [`BtrBlocksCompressorBuilder`] that [`WriteStrategyBuilder::build`] will finalize.
    /// `IntDictScheme` is automatically excluded from the data compressor to prevent recursive
    /// dictionary encoding.
    BtrBlocks(BtrBlocksCompressorBuilder),
    /// An opaque compressor used as-is for both data and stats compression.
    Opaque(Arc<dyn CompressorPlugin>),
}

/// Build a new [writer strategy](LayoutStrategy) to compress and reorganize chunks of a Vortex
/// file.
///
/// Vortex provides an out-of-the-box file writer that optimizes the layout of chunks on-disk,
/// repartitioning and compressing them to strike a balance between size on-disk,
/// bulk decoding performance, and IOPS required to perform an indexed read.
pub struct WriteStrategyBuilder {
    compressor: CompressorConfig,
    row_block_size: usize,
    field_writers: HashMap<FieldPath, Arc<dyn LayoutStrategy>>,
    field_compressors: HashMap<FieldPath, Arc<dyn CompressorPlugin>>,
    allow_encodings: Option<HashSet<ArrayId>>,
    flat_strategy: Option<Arc<dyn LayoutStrategy>>,
    /// Optional override for the per-chunk stats that
    /// `CompressingStrategy` pre-computes before each compression
    /// call. `None` keeps the default of [`Stat::all()`]; callers
    /// that know their scheme set never reads certain stats can
    /// narrow this to skip the corresponding per-fragment scans.
    compressing_stats: Option<Arc<[vortex_array::expr::stats::Stat]>>,
}

impl Default for WriteStrategyBuilder {
    /// Create a new empty builder. It can be further configured,
    /// and then finally built yielding the [`LayoutStrategy`].
    fn default() -> Self {
        Self {
            compressor: CompressorConfig::BtrBlocks(BtrBlocksCompressorBuilder::default()),
            row_block_size: 8192,
            field_writers: HashMap::new(),
            field_compressors: HashMap::new(),
            allow_encodings: Some(ALLOWED_ENCODINGS.clone()),
            flat_strategy: None,
            compressing_stats: None,
        }
    }
}

impl WriteStrategyBuilder {
    /// Override the row block size used to determine the zone map sizes.
    pub fn with_row_block_size(mut self, row_block_size: usize) -> Self {
        self.row_block_size = row_block_size;
        self
    }

    /// Override the default write layout for a specific field somewhere in the nested
    /// schema tree.
    pub fn with_field_writer(
        mut self,
        field: impl Into<FieldPath>,
        writer: Arc<dyn LayoutStrategy>,
    ) -> Self {
        self.field_writers.insert(field.into(), writer);
        self
    }

    /// Override the allowed array encodings for normalization.
    pub fn with_allow_encodings(mut self, allow_encodings: HashSet<ArrayId>) -> Self {
        self.allow_encodings = Some(allow_encodings);
        self
    }

    /// Override the flat layout strategy used for leaf chunks.
    ///
    /// By default, this uses [`FlatLayoutStrategy`]. This can be used to substitute a custom
    /// layout strategy, e.g. one that inlines constant array buffers for GPU reads.
    pub fn with_flat_strategy(mut self, flat: Arc<dyn LayoutStrategy>) -> Self {
        self.flat_strategy = Some(flat);
        self
    }

    /// Override the default [`BtrBlocksCompressorBuilder`] used for compression.
    ///
    /// The builder is finalized during [`build`](Self::build), producing two compressors: one for
    /// data (with `IntDictScheme` excluded) and one for stats.
    pub fn with_btrblocks_builder(mut self, builder: BtrBlocksCompressorBuilder) -> Self {
        self.compressor = CompressorConfig::BtrBlocks(builder);
        self
    }

    /// Set the compressor to an opaque [`CompressorPlugin`].
    ///
    /// The compressor is used as-is for both data and stats compression.
    pub fn with_compressor<C: CompressorPlugin>(mut self, compressor: C) -> Self {
        self.compressor = CompressorConfig::Opaque(Arc::new(compressor));
        self
    }

    /// Override the per-chunk stats that the leaf `CompressingStrategy`
    /// pre-computes before each compression call.
    ///
    /// Defaults to [`Stat::all()`](vortex_array::expr::stats::Stat::all),
    /// which costs a per-fragment scan for every stat in the enum (some
    /// share a single scan, some don't). Callers whose scheme set
    /// reads only a subset can narrow this to skip the
    /// `IsSorted`/`IsStrictSorted`/`UncompressedSizeInBytes`/`Sum`/
    /// `NaNCount` scans (or any other unused stat). The narrowed set
    /// is shared by both the fallback chain and every per-field
    /// override.
    pub fn with_compressing_stats(
        mut self,
        stats: impl IntoIterator<Item = vortex_array::expr::stats::Stat>,
    ) -> Self {
        self.compressing_stats = Some(stats.into_iter().collect());
        self
    }

    /// Register per-leaf-field [`CompressorPlugin`] overrides.
    ///
    /// Each entry instructs [`Self::build`] to construct a per-field
    /// `LayoutStrategy` chain that mirrors the default fallback shape
    /// (repartition → zoned-stats → dict-or-fallback → coalescing →
    /// compressing → buffered → chunked → flat) but swaps the leaf
    /// [`CompressingStrategy`]'s `CompressorPlugin` for the registered
    /// override. The resulting strategies are inserted into
    /// `field_writers`, taking precedence over any
    /// [`Self::with_field_writer`] entry for the same `FieldPath`.
    ///
    /// Useful for callers that want per-column compressor state (e.g.
    /// a per-column scheme-selection-winner cache) without
    /// reconstructing the rest of the strategy chain.
    pub fn with_field_compressors<I>(mut self, compressors: I) -> Self
    where
        I: IntoIterator<Item = (FieldPath, Arc<dyn CompressorPlugin>)>,
    {
        self.field_compressors = compressors.into_iter().collect();
        self
    }

    /// Builds the canonical [`LayoutStrategy`] implementation, with the configured overrides
    /// applied.
    pub fn build(self) -> Arc<dyn LayoutStrategy> {
        let flat: Arc<dyn LayoutStrategy> = if let Some(flat) = self.flat_strategy {
            flat
        } else if let Some(allow_encodings) = self.allow_encodings {
            Arc::new(FlatLayoutStrategy::default().with_allow_encodings(allow_encodings))
        } else {
            Arc::new(FlatLayoutStrategy::default())
        };

        // 5. compress each chunk.
        // Exclude IntDictScheme from the data compressor because DictStrategy (step 3) already
        // dictionary-encodes columns. Allowing IntDictScheme here would redundantly
        // dictionary-encode the integer codes produced by that earlier step.
        let default_data_compressor: Arc<dyn CompressorPlugin> = match &self.compressor {
            CompressorConfig::BtrBlocks(builder) => Arc::new(
                builder
                    .clone()
                    .exclude_schemes([IntDictScheme.id()])
                    .build(),
            ),
            CompressorConfig::Opaque(compressor) => Arc::clone(compressor),
        };

        // 2.1. | 3.1. compress stats tables and dict values.
        let stats_compressor: Arc<dyn CompressorPlugin> = match self.compressor {
            CompressorConfig::BtrBlocks(builder) => Arc::new(builder.build()),
            CompressorConfig::Opaque(compressor) => compressor,
        };
        let compress_then_flat = CompressingStrategy::new(Arc::clone(&flat), stats_compressor);

        // Build the leaf-field strategy chain (steps 1–7 from the original
        // composition) parameterised on which `CompressorPlugin` to plug
        // into the leaf `CompressingStrategy`. The chain is identical
        // shape for every field; only the data compressor varies.
        let compressing_stats = self.compressing_stats.clone();
        let build_field_chain = |data_compressor: Arc<dyn CompressorPlugin>| -> Arc<dyn LayoutStrategy> {
            // 7. for each chunk create a flat layout
            let chunked = ChunkedLayoutStrategy::new(Arc::clone(&flat));
            // 6. buffer chunks so they end up with closer segment ids physically
            let buffered = BufferedStrategy::new(chunked, 2 * ONE_MEG); // 2MB

            let mut compressing = CompressingStrategy::new(buffered, Arc::clone(&data_compressor));
            if let Some(stats) = compressing_stats.as_deref() {
                compressing = compressing.with_stats(stats);
            }

            // 4. prior to compression, coalesce up to a minimum size
            let coalescing = RepartitionStrategy::new(
                compressing,
                RepartitionWriterOptions {
                    block_size_minimum: ONE_MEG,
                    block_len_multiple: self.row_block_size,
                    block_size_target: Some(ONE_MEG),
                    canonicalize: true,
                },
            );

            // 3. apply dict encoding or fallback
            //
            // The dict probe (DictStrategy's first-chunk check) reuses the data_compressor
            // so any pre-trained codec state (e.g. an FSSTSchemeWithPretrained variant
            // installed via BtrBlocksCompressorBuilder::with_new_scheme_arc) flows into the
            // probe instead of being silently replaced with a stock BtrBlocksCompressor that
            // re-trains FSST symbol tables on every probe. Without this, the probe runs
            // FSST training per chunk on every string column even when the data path is
            // configured to skip training — wiping out the streaming-ingest win.
            let dict = DictStrategy::new(
                coalescing.clone(),
                compress_then_flat.clone(),
                coalescing,
                Default::default(),
            )
            .with_probe_compressor(Arc::clone(&data_compressor));

            // 2. calculate stats for each row group
            let stats = ZonedStrategy::new(
                dict,
                compress_then_flat.clone(),
                ZonedLayoutOptions {
                    block_size: self.row_block_size,
                    ..Default::default()
                },
            );

            // 1. repartition each column to fixed row counts
            let repartition = RepartitionStrategy::new(
                stats,
                RepartitionWriterOptions {
                    // No minimum block size in bytes
                    block_size_minimum: 0,
                    // Always repartition into 8K row blocks
                    block_len_multiple: self.row_block_size,
                    block_size_target: None,
                    canonicalize: false,
                },
            );
            Arc::new(repartition)
        };

        // Convert per-field compressor overrides into per-field
        // `LayoutStrategy` entries with the shared chain shape, then
        // merge them into any explicit `with_field_writer` overrides
        // (explicit `with_field_writer` takes precedence).
        let mut field_writers = self.field_writers;
        for (path, compressor) in self.field_compressors {
            field_writers
                .entry(path)
                .or_insert_with(|| build_field_chain(compressor));
        }

        let fallback = build_field_chain(default_data_compressor);

        // 0. start with splitting columns
        let validity_strategy = CollectStrategy::new(compress_then_flat);

        // Take any field overrides from the builder and apply them to the final strategy.
        let table_strategy =
            TableStrategy::new(Arc::new(validity_strategy), fallback).with_field_writers(field_writers);

        Arc::new(table_strategy)
    }
}
