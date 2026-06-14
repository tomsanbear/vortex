// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Frame of Reference integer encoding.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_error::VortexResult;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;

use super::BitPackingScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// Frame of Reference encoding.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FoRScheme;

impl Scheme for FoRScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.for"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    /// Dict codes always start at 0, so FoR (which subtracts the min) is a no-op.
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        vec![
            AncestorExclusion {
                ancestor: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: FloatDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: StringDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: BinaryDictScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // FoR only subtracts the min. Without further compression (e.g. BitPacking), the output is
        // the same size.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        // Read `min` and `max` directly from the array's stats cache
        // rather than triggering the full `IntegerStats` compute via
        // `data.integer_stats`: the cache is populated by
        // `CompressingStrategy`'s `compute_all(&Stat::all(), ...)`
        // before any scheme's estimate runs, so this is `O(1)` on a
        // cache hit. FoR's own `compress` doesn't read `IntegerStats`
        // either, so paying the full compute here is pure waste on
        // the freeze fast path that the C-prime per-column cache is
        // supposed to accelerate. Mirrors the bitpacking gate's
        // direct-stats-read pattern.
        let primitive = data.array_as_primitive();
        let array_ref = primitive.as_ref();
        let full_width = primitive.ptype().bit_width() as u32;
        #[allow(unused_comparisons, clippy::absurd_extreme_comparisons)]
        let (min_is_zero, min_is_negative, max_minus_min, max_value_u128) =
            vortex_array::match_each_integer_ptype!(primitive.ptype(), |P| {
                let stats_set = array_ref.statistics();
                let min = stats_set.compute_min::<P>(exec_ctx).unwrap_or_default();
                let max = stats_set.compute_max::<P>(exec_ctx).unwrap_or_default();
                let min_i128 = min as i128;
                let max_i128 = max as i128;
                let diff = (max_i128 - min_i128) as u128;
                (min == 0, min < 0, diff, max_i128 as u128)
            });

        // Only apply when the min is not already zero.
        if min_is_zero {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Difference between max and min.
        let for_bitwidth = match max_minus_min.checked_ilog2() {
            Some(l) => l + 1,
            // If max-min == 0, the we should be compressing this as a constant array.
            None => return CompressionEstimate::Verdict(EstimateVerdict::Skip),
        };

        // For signed integer inputs whose (max - min) span exceeds the
        // signed type's positive range, `FoR.encode`'s `wrapping_sub`
        // produces biased values that read as negative when interpreted
        // as the original signed type. The unconditional
        // `BitPackingScheme.compress` call inside `Self::compress`
        // (immediately below) then trips `bitpack_encode`'s
        // negative-integer guard. Skip FoR for these inputs — the
        // compression ratio would have been 1.0 in this regime anyway
        // (`for_bitwidth == full_width`), so refusing here costs no
        // realistic compression while keeping the bitpack precondition
        // intact for the cascade's other consumers.
        let signed_full_width = full_width.saturating_sub(1);
        if primitive.ptype().is_signed_int() && for_bitwidth > signed_full_width {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // If BitPacking can be applied (only non-negative values) and FoR doesn't reduce bit width
        // compared to BitPacking, don't use FoR since it has a small amount of overhead (storing
        // the reference) for effectively no benefits. Only consult when
        // min >= 0 because BitPacking can't be applied without ZigZag
        // otherwise.
        if !min_is_negative {
            if let Some(max_log) = max_value_u128.checked_ilog2() {
                let bitpack_bitwidth = max_log + 1;
                if for_bitwidth >= bitpack_bitwidth {
                    return CompressionEstimate::Verdict(EstimateVerdict::Skip);
                }
            }
        }

        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            full_width as f64 / for_bitwidth as f64,
        ))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
        let for_array = FoR::encode(primitive)?;
        let biased = for_array
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?;

        // Immediately bitpack. If any other scheme was preferable, it would be chosen instead
        // of bitpacking.
        // NOTE: we could delegate in the future if we had another downstream codec that performs
        //  as well.
        let leaf_ctx = compress_ctx.clone().as_leaf();
        let biased_data =
            ArrayAndStats::new(biased.into_array(), compress_ctx.merged_stats_options());
        let compressed = BitPackingScheme.compress(compressor, &biased_data, leaf_ctx, exec_ctx)?;

        // TODO(connor): This should really be `new_unchecked`.
        let for_compressed = FoR::try_new(compressed, for_array.reference_scalar().clone())?;
        for_compressed
            .as_ref()
            .statistics()
            .inherit_from(for_array.as_ref().statistics());

        Ok(for_compressed.into_array())
    }
}
