// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST with a caller-provided pretrained symbol table.
//!
//! [`FSSTSchemeWithPretrained`] is a drop-in replacement for the default
//! [`FSSTScheme`](super::FSSTScheme) that skips the per-array symbol-table
//! training step. Callers obtain a `fsst::Compressor` once (typically by
//! training on a sample of representative data from the same column, or
//! by reusing one learned during an earlier fragment write) and inject
//! it into the cascading compressor via
//! [`BtrBlocksCompressorBuilder::with_new_scheme_arc`](crate::BtrBlocksCompressorBuilder::with_new_scheme_arc),
//! after first excluding the default [`FSSTScheme`](super::FSSTScheme) by
//! its [`SchemeId`](vortex_compressor::scheme::SchemeId) so the registered
//! scheme set never holds two schemes with the same id.
//!
//! Why this matters: the default `FSSTScheme::compress` calls
//! `fsst_train_compressor` on every array it sees. With
//! `CompressionEstimate::Deferred(Sample)`, the framework first runs a
//! sample compress to estimate the ratio (one train) and, if FSST wins,
//! runs the full-array compress (a second train). Both trainings are
//! O(sample / array size) but carry substantial fixed cost — they
//! dominate write-side CPU on string-heavy streaming-ingest workloads
//! that produce many small fragments. A pretrained variant elides both
//! trainings and reuses the supplied symbol table, leaving only the
//! data-linear `fsst_compress` work on the hot path.
//!
//! Correctness: an `fsst::Compressor` is lossless for ANY input because
//! bytes not covered by the trained symbol table are emitted as
//! `[ESCAPE_CODE, raw_byte]`. Reusing a symbol table on a column whose
//! distribution has drifted is therefore correct; the only failure mode
//! is degraded compression ratio. The cascading selector observes the
//! sample-compressed size and will pick a different scheme if FSST no
//! longer wins on the new distribution, so silent regressions still
//! surface through normal scheme selection.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_error::VortexResult;
use vortex_fsst::FSST;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::fsst_compress;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::string::FSSTScheme;

/// FSST with a caller-provided pretrained symbol table.
///
/// Shares its [`scheme_name`](Scheme::scheme_name) — and therefore
/// [`SchemeId`](vortex_compressor::scheme::SchemeId) — with the default
/// [`FSSTScheme`], so the cascading compressor's exclusion rules,
/// cascade-history tracking, and on-disk array encoding identity are
/// preserved. The builder enforces single-scheme-per-id via an assertion
/// in [`with_new_scheme_arc`](crate::BtrBlocksCompressorBuilder::with_new_scheme_arc),
/// so callers must call
/// [`exclude_schemes`](crate::BtrBlocksCompressorBuilder::exclude_schemes)
/// with `FSSTScheme.id()` before registering this variant.
#[derive(Clone)]
pub struct FSSTSchemeWithPretrained {
    /// The pretrained symbol-table compressor reused for every
    /// `compress` invocation (both the framework's sample-compress for
    /// estimation and the post-selection full compress).
    pretrained: Arc<fsst::Compressor>,
}

impl std::fmt::Debug for FSSTSchemeWithPretrained {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `fsst::Compressor` does not implement `Debug` and its full state
        // (the 256-entry symbol table) is not meaningful in trace output.
        // Emit a stable opaque marker so tracing remains useful without
        // leaking compressor internals.
        f.debug_struct("FSSTSchemeWithPretrained")
            .field("pretrained", &"<fsst::Compressor>")
            .finish()
    }
}

impl FSSTSchemeWithPretrained {
    /// Construct an FSST scheme that always compresses with `pretrained`
    /// instead of training a fresh symbol table per array.
    ///
    /// The `Arc<fsst::Compressor>` is shared verbatim across calls; the
    /// caller chooses its scope (per-session, per-column, per-table).
    pub fn new(pretrained: Arc<fsst::Compressor>) -> Self {
        Self { pretrained }
    }

    /// Borrow the underlying pretrained compressor. Useful for callers
    /// that share the same symbol table across multiple builder
    /// constructions.
    pub fn compressor(&self) -> &Arc<fsst::Compressor> {
        &self.pretrained
    }
}

impl Scheme for FSSTSchemeWithPretrained {
    fn scheme_name(&self) -> &'static str {
        FSSTScheme.scheme_name()
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        FSSTScheme.matches(canonical)
    }

    fn num_children(&self) -> usize {
        FSSTScheme.num_children()
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // Match the default FSSTScheme estimate-shape so the cascading
        // selector still runs a sample-compress for ratio estimation —
        // this scheme is still chosen by its on-data ratio, just without
        // the training cost on either the sample or the full array.
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let utf8 = data.array_as_varbinview().into_owned();
        // The single difference from `FSSTScheme::compress`: instead of
        // `fsst_train_compressor(&utf8)`, reuse `self.pretrained`. The
        // resulting `FSSTArray` is byte-identical in shape — same
        // symbol-table buffers, same encoded codes — so downstream
        // decode paths require no changes.
        let fsst = fsst_compress(&utf8, utf8.len(), utf8.dtype(), &self.pretrained, exec_ctx);

        let uncompressed_lengths_primitive = fsst
            .uncompressed_lengths()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let compressed_original_lengths = compressor.compress_child(
            &uncompressed_lengths_primitive.into_array(),
            &compress_ctx,
            self.id(),
            0,
            exec_ctx,
        )?;

        let codes_offsets_primitive = fsst
            .codes()
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let compressed_codes_offsets = compressor.compress_child(
            &codes_offsets_primitive.into_array(),
            &compress_ctx,
            self.id(),
            1,
            exec_ctx,
        )?;
        let compressed_codes = VarBinArray::try_new(
            compressed_codes_offsets,
            fsst.codes().bytes().clone(),
            fsst.codes().dtype().clone(),
            fsst.codes().validity()?,
        )?;

        let fsst = FSST::try_new(
            fsst.dtype().clone(),
            fsst.symbols().clone(),
            fsst.symbol_lengths().clone(),
            compressed_codes,
            compressed_original_lengths,
            exec_ctx,
        )?;

        Ok(fsst.into_array())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::session::ArraySession;
    use vortex_compressor::scheme::SchemeExt;
    use vortex_error::VortexResult;
    use vortex_fsst::fsst_train_compressor;
    use vortex_session::VortexSession;

    use super::FSSTSchemeWithPretrained;
    use crate::BtrBlocksCompressorBuilder;
    use crate::schemes::string::FSSTScheme;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    /// The pretrained variant carries the same [`SchemeId`] as the default
    /// [`FSSTScheme`], so a caller may swap it in via
    /// `exclude_schemes` + `with_new_scheme_arc` and the cascade's
    /// scheme-id-based invariants (exclusion lookup, history tracking)
    /// still resolve correctly.
    #[test]
    fn shares_scheme_id_with_default_fsst() {
        let utf8 = VarBinViewArray::from_iter(
            (0..16).map(|i| Some(format!("hello world {i}"))),
            DType::Utf8(Nullability::NonNullable),
        );
        let utf8_for_train = utf8.clone();
        let pretrained = Arc::new(fsst_train_compressor(&utf8_for_train));

        let pretrained_scheme = FSSTSchemeWithPretrained::new(pretrained);

        assert_eq!(pretrained_scheme.id(), FSSTScheme.id());
    }

    /// Round-trip: writing with the pretrained variant produces a
    /// `FSSTArray` whose decoded payload matches the original input.
    #[test]
    fn pretrained_roundtrip_matches_input() -> VortexResult<()> {
        let values: Vec<String> = (0..256)
            .map(|i| format!("pretrained sample row {i:04} payload"))
            .collect();
        let input = VarBinViewArray::from_iter(
            values.iter().map(|s| Some(s.as_str())),
            DType::Utf8(Nullability::NonNullable),
        );

        let pretrained = Arc::new(fsst_train_compressor(&input.clone()));

        let compressor = BtrBlocksCompressorBuilder::default()
            .exclude_schemes([FSSTScheme.id()])
            .with_new_scheme_arc(Arc::new(FSSTSchemeWithPretrained::new(pretrained)))
            .build();

        let compressed = compressor.compress(
            &input.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )?;

        assert_arrays_eq!(compressed, input);
        Ok(())
    }
}
