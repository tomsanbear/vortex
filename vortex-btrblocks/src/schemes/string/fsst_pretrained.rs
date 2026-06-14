// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST compression with a caller-provided pretrained symbol table.
//!
//! The default [`FSSTScheme`](super::FSSTScheme) trains a fresh symbol table on
//! every array (twice per array due to `Deferred(Sample)` estimation).
//! [`FSSTSchemeWithPretrained`] reuses a caller-supplied `fsst::Compressor`,
//! eliding both trainings.
//!
//! `fsst::Compressor` is lossless for any input via the `ESCAPE_CODE` fallback,
//! so reusing a symbol table on drifted data is correct — only ratio degrades.
//! The cascading selector deselects FSST if it stops winning.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::string::FSSTScheme;
use crate::schemes::string::fsst::fsst_compress_with_compressor;

/// FSST compression with a caller-provided pretrained symbol table.
///
/// Shares [`SchemeId`](vortex_compressor::scheme::SchemeId) with the default
/// [`FSSTScheme`], preserving cascade exclusion rules and on-disk encoding
/// identity. Register via
/// [`replace_scheme_arc`](crate::BtrBlocksCompressorBuilder::replace_scheme_arc).
#[derive(Clone)]
pub struct FSSTSchemeWithPretrained {
    pretrained: Arc<fsst::Compressor>,
}

impl std::fmt::Debug for FSSTSchemeWithPretrained {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FSSTSchemeWithPretrained")
            .field("pretrained", &"<fsst::Compressor>")
            .finish()
    }
}

impl FSSTSchemeWithPretrained {
    /// Creates a new pretrained FSST scheme.
    pub fn new(pretrained: Arc<fsst::Compressor>) -> Self {
        Self { pretrained }
    }

    /// Borrow the underlying pretrained compressor.
    pub fn compressor(&self) -> &Arc<fsst::Compressor> {
        &self.pretrained
    }
}

impl Scheme for FSSTSchemeWithPretrained {
    fn scheme_name(&self) -> &'static str {
        // Shared SchemeId preserves cascade exclusion rules and on-disk identity.
        FSSTScheme.scheme_name()
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        FSSTScheme.matches(canonical)
    }

    /// Children: lengths=0, code_offsets=1.
    fn num_children(&self) -> usize {
        FSSTScheme.num_children()
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        fsst_compress_with_compressor(
            &self.pretrained,
            compressor,
            data,
            compress_ctx,
            self.id(),
            exec_ctx,
        )
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
    use vortex_fsst::FSST as FsstEncoding;
    use vortex_fsst::fsst_train_compressor;
    use vortex_session::VortexSession;

    use super::FSSTSchemeWithPretrained;
    use crate::BtrBlocksCompressorBuilder;
    use crate::schemes::string::FSSTScheme;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn build_pretrained_compressor(values: &[String]) -> (VarBinViewArray, Arc<fsst::Compressor>) {
        let array = VarBinViewArray::from_iter(
            values.iter().map(|s| Some(s.as_str())),
            DType::Utf8(Nullability::NonNullable),
        );
        let pretrained = Arc::new(fsst_train_compressor(&array));
        (array, pretrained)
    }

    #[test]
    fn shares_scheme_id_with_default_fsst() {
        let values: Vec<String> = (0..16).map(|i| format!("hello world {i}")).collect();
        let (_, pretrained) = build_pretrained_compressor(&values);

        let pretrained_scheme = FSSTSchemeWithPretrained::new(pretrained);
        assert_eq!(pretrained_scheme.id(), FSSTScheme.id());
    }

    #[test]
    fn pretrained_roundtrip_matches_input() -> VortexResult<()> {
        let values: Vec<String> = (0..256)
            .map(|i| format!("pretrained sample row {i:04} payload"))
            .collect();
        let (input, pretrained) = build_pretrained_compressor(&values);

        let compressor = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme_arc(Arc::new(FSSTSchemeWithPretrained::new(pretrained)))
            .build();

        let compressed = compressor.compress(
            &input.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )?;

        assert!(
            compressed.as_opt::<FsstEncoding>().is_some(),
            "expected FSST encoding, got {}",
            compressed.encoding_id()
        );
        assert_arrays_eq!(compressed, input);
        Ok(())
    }

    /// Partially overlapping corpora: the non-shared bytes exercise the
    /// ESCAPE_CODE fallback while the shared substrings still compress.
    #[test]
    fn pretrained_roundtrip_with_drifted_data() -> VortexResult<()> {
        let training_corpus: Vec<String> = (0..1000)
            .map(|i| format!("https://api.example.com/v2/orders/{i:08x}/status"))
            .collect();
        let (_, pretrained) = build_pretrained_compressor(&training_corpus);

        let compress_corpus: Vec<String> = (0..1000)
            .map(|i| format!("https://cdn.example.com/v2/images/{i:08x}/thumbnail"))
            .collect();
        let input = VarBinViewArray::from_iter(
            compress_corpus.iter().map(|s| Some(s.as_str())),
            DType::Utf8(Nullability::NonNullable),
        );

        let compressor = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme_arc(Arc::new(FSSTSchemeWithPretrained::new(Arc::clone(
                &pretrained,
            ))))
            .build();

        let compressed = compressor.compress(
            &input.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )?;

        assert!(
            compressed.as_opt::<FsstEncoding>().is_some(),
            "expected FSST encoding even on drifted data, got {}",
            compressed.encoding_id()
        );
        let fsst_array = compressed.as_opt::<FsstEncoding>().unwrap();
        assert_eq!(
            fsst_array.symbols().as_slice(),
            pretrained.symbol_table(),
            "should use the pretrained symbol table, not train a new one"
        );
        assert_arrays_eq!(compressed, input);
        Ok(())
    }

    #[test]
    fn pretrained_wins_against_default_schemes() -> VortexResult<()> {
        let values: Vec<String> = (0..1000)
            .map(|i| {
                format!(
                    "this_is_a_common_prefix_with_some_variation_{i}_and_a_common_suffix_pattern"
                )
            })
            .collect();
        let (input, pretrained) = build_pretrained_compressor(&values);

        let compressor = BtrBlocksCompressorBuilder::default()
            .exclude_schemes([FSSTScheme.id()])
            .with_new_scheme_arc(Arc::new(FSSTSchemeWithPretrained::new(pretrained)))
            .build();

        let compressed = compressor.compress(
            &input.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )?;

        assert!(
            compressed.as_opt::<FsstEncoding>().is_some(),
            "expected FSST encoding when competing against default schemes, got {}",
            compressed.encoding_id()
        );
        assert_arrays_eq!(compressed, input);
        Ok(())
    }

    #[test]
    fn pretrained_roundtrip_nullable() -> VortexResult<()> {
        let values: Vec<Option<String>> = (0..256)
            .map(|i| {
                if i % 5 == 0 {
                    None
                } else {
                    Some(format!("nullable pretrained row {i:04} payload"))
                }
            })
            .collect();
        let input = VarBinViewArray::from_iter(
            values.iter().map(|s| s.as_deref()),
            DType::Utf8(Nullability::Nullable),
        );
        let non_null_input = VarBinViewArray::from_iter(
            values.iter().filter_map(|s| s.as_deref()).map(Some),
            DType::Utf8(Nullability::NonNullable),
        );
        let pretrained = Arc::new(fsst_train_compressor(&non_null_input));

        let compressor = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme_arc(Arc::new(FSSTSchemeWithPretrained::new(pretrained)))
            .build();

        let compressed = compressor.compress(
            &input.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )?;

        assert_arrays_eq!(compressed, input);
        Ok(())
    }

    #[test]
    fn pretrained_uses_supplied_symbol_table() -> VortexResult<()> {
        let training_values: Vec<String> = (0..256)
            .map(|i| format!("training-data-{i:04}-with-pattern"))
            .collect();
        let (_, pretrained) = build_pretrained_compressor(&training_values);

        let compress_values: Vec<String> = (0..256)
            .map(|i| format!("different-input-{i:04}-other-pattern"))
            .collect();
        let input = VarBinViewArray::from_iter(
            compress_values.iter().map(|s| Some(s.as_str())),
            DType::Utf8(Nullability::NonNullable),
        );

        let compressor = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme_arc(Arc::new(FSSTSchemeWithPretrained::new(Arc::clone(
                &pretrained,
            ))))
            .build();

        let compressed =
            compressor.compress(&input.into_array(), &mut SESSION.create_execution_ctx())?;

        let fsst_array = compressed
            .as_opt::<FsstEncoding>()
            .expect("expected FSST encoding");
        assert_eq!(fsst_array.symbols().as_slice(), pretrained.symbol_table(),);
        assert_eq!(
            fsst_array.symbol_lengths().as_slice(),
            pretrained.symbol_lengths(),
        );
        Ok(())
    }

    #[test]
    #[should_panic(expected = "already present")]
    fn builder_rejects_duplicate_scheme_id() {
        let values: Vec<String> = (0..16).map(|i| format!("row {i}")).collect();
        let (_, pretrained) = build_pretrained_compressor(&values);

        let _compressor = BtrBlocksCompressorBuilder::default()
            .with_new_scheme_arc(Arc::new(FSSTSchemeWithPretrained::new(pretrained)))
            .build();
    }
}
