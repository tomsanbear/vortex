// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::NumCast;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::dtype::UnsignedPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::{Buffer, BufferMut};
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::RunEnd;
use crate::array::RunEndArrayExt;

const SORTED_LINEAR_RUNS_PER_INDEX_THRESHOLD: usize = 16;
const UNSORTED_LINEAR_RUNS_PER_INDEX_THRESHOLD: usize = 4;

impl TakeExecute for RunEnd {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "index cast to usize inside macro"
    )]
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let primitive_indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let indices_validity = primitive_indices.validity()?;
        let indices_mask = indices_validity.execute_mask(primitive_indices.len(), ctx)?;

        let checked_indices = match_each_integer_ptype!(primitive_indices.ptype(), |P| {
            primitive_indices
                .as_slice::<P>()
                .iter()
                .copied()
                .enumerate()
                .map(|(idx_pos, idx)| {
                    if !indices_mask.value(idx_pos) {
                        return Ok(0);
                    }

                    let usize_idx = idx as usize;
                    if usize_idx >= array.len() {
                        vortex_bail!(OutOfBounds: usize_idx, 0, array.len());
                    }
                    Ok(usize_idx)
                })
                .collect::<VortexResult<Vec<_>>>()?
        });

        take_indices_unchecked_with_mask(
            array,
            &checked_indices,
            &indices_validity,
            &indices_mask,
            ctx,
        )
        .map(Some)
    }
}

/// Perform a take operation on a RunEndArray.
pub fn take_indices_unchecked<T: AsPrimitive<usize>>(
    array: ArrayView<'_, RunEnd>,
    indices: &[T],
    validity: &Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let validity_mask = validity.execute_mask(indices.len(), ctx)?;
    take_indices_unchecked_with_mask(array, indices, validity, &validity_mask, ctx)
}

fn take_indices_unchecked_with_mask<T: AsPrimitive<usize>>(
    array: ArrayView<'_, RunEnd>,
    indices: &[T],
    validity: &Validity,
    validity_mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ends = array.ends().clone().execute::<PrimitiveArray>(ctx)?;

    let physical_indices = match_each_unsigned_integer_ptype!(ends.ptype(), |I| {
        let end_slices = ends.as_slice::<I>();
        let physical_indices =
            physical_indices(end_slices, array.offset(), indices, validity_mask);

        PrimitiveArray::new(physical_indices, validity.clone())
    });

    array.values().take(physical_indices.into_array())
}

fn physical_indices<I, T>(
    ends: &[I],
    offset: usize,
    indices: &[T],
    validity_mask: &Mask,
) -> Buffer<u64>
where
    I: UnsignedPType,
    T: AsPrimitive<usize>,
{
    let (valid_count, valid_indices_sorted) = valid_indices_stats(indices, validity_mask);

    if valid_count == 0 {
        return Buffer::zeroed(indices.len());
    }

    if valid_indices_sorted
        && prefer_linear_scan(
            ends.len(),
            valid_count,
            SORTED_LINEAR_RUNS_PER_INDEX_THRESHOLD,
        )
    {
        return physical_indices_linear_sorted(ends, offset, indices, validity_mask);
    }

    if prefer_linear_scan(
        ends.len(),
        valid_count,
        UNSORTED_LINEAR_RUNS_PER_INDEX_THRESHOLD,
    ) {
        return physical_indices_linear_unsorted(ends, offset, indices, validity_mask, valid_count);
    }

    physical_indices_binary(ends, offset, indices, validity_mask)
}

fn valid_indices_stats<T: AsPrimitive<usize>>(
    indices: &[T],
    validity_mask: &Mask,
) -> (usize, bool) {
    let mut valid_count = 0;
    let mut previous_idx = None;
    let mut sorted = true;

    for (idx_pos, idx) in indices.iter().enumerate() {
        if !validity_mask.value(idx_pos) {
            continue;
        }

        let idx = idx.as_();
        if previous_idx.is_some_and(|previous_idx| previous_idx > idx) {
            sorted = false;
        }
        previous_idx = Some(idx);
        valid_count += 1;
    }

    (valid_count, sorted)
}

fn prefer_linear_scan(
    ends_len: usize,
    valid_count: usize,
    runs_per_index_threshold: usize,
) -> bool {
    ends_len <= valid_count.saturating_mul(runs_per_index_threshold)
}

fn physical_indices_linear_sorted<I, T>(
    ends: &[I],
    offset: usize,
    indices: &[T],
    validity_mask: &Mask,
) -> Buffer<u64>
where
    I: UnsignedPType,
    T: AsPrimitive<usize>,
{
    let mut physical_indices = BufferMut::zeroed(indices.len());
    let mut run_idx = 0;

    for (idx_pos, idx) in indices.iter().enumerate() {
        if !validity_mask.value(idx_pos) {
            continue;
        }

        let logical_idx = idx.as_() + offset;
        advance_run(ends, &mut run_idx, logical_idx);
        physical_indices[idx_pos] = run_idx as u64;
    }

    physical_indices.freeze()
}

fn physical_indices_linear_unsorted<I, T>(
    ends: &[I],
    offset: usize,
    indices: &[T],
    validity_mask: &Mask,
    valid_count: usize,
) -> Buffer<u64>
where
    I: UnsignedPType,
    T: AsPrimitive<usize>,
{
    let mut pairs = Vec::with_capacity(valid_count);
    for (idx_pos, idx) in indices.iter().enumerate() {
        if validity_mask.value(idx_pos) {
            pairs.push((idx.as_(), idx_pos));
        }
    }
    pairs.sort_unstable();

    let mut physical_indices = BufferMut::zeroed(indices.len());
    let mut run_idx = 0;

    for (idx, idx_pos) in pairs {
        let logical_idx = idx + offset;
        advance_run(ends, &mut run_idx, logical_idx);
        physical_indices[idx_pos] = run_idx as u64;
    }

    physical_indices.freeze()
}

fn physical_indices_binary<I, T>(
    ends: &[I],
    offset: usize,
    indices: &[T],
    validity_mask: &Mask,
) -> Buffer<u64>
where
    I: UnsignedPType,
    T: AsPrimitive<usize>,
{
    let mut physical_indices = BufferMut::zeroed(indices.len());

    for (idx_pos, idx) in indices.iter().enumerate() {
        if !validity_mask.value(idx_pos) {
            continue;
        }

        let logical_idx = idx.as_() + offset;
        physical_indices[idx_pos] = physical_index_binary(ends, logical_idx) as u64;
    }

    physical_indices.freeze()
}

fn physical_index_binary<I: UnsignedPType>(ends: &[I], logical_idx: usize) -> usize {
    let index = match <I as NumCast>::from(logical_idx) {
        Some(logical_idx) => ends.partition_point(|end| *end <= logical_idx),
        None => ends.len(),
    };
    index.min(ends.len() - 1)
}

fn advance_run<I: UnsignedPType>(ends: &[I], run_idx: &mut usize, logical_idx: usize) {
    while *run_idx + 1 < ends.len() && run_end_le_logical_idx(ends[*run_idx], logical_idx) {
        *run_idx += 1;
    }
}

fn run_end_le_logical_idx<I: UnsignedPType>(run_end: I, logical_idx: usize) -> bool {
    match <I as NumCast>::from(logical_idx) {
        Some(logical_idx) => run_end <= logical_idx,
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::RunEnd;
    use crate::RunEndArray;

    fn ree_array() -> RunEndArray {
        RunEnd::encode(
            buffer![1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5].into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap()
    }

    #[test]
    fn ree_take() {
        let taken = ree_array().take(buffer![9, 8, 1, 3].into_array()).unwrap();
        let expected = PrimitiveArray::from_iter(vec![5i32, 5, 1, 4]).into_array();
        assert_arrays_eq!(taken, expected);
    }

    #[test]
    fn ree_take_end() {
        let taken = ree_array().take(buffer![11].into_array()).unwrap();
        let expected = PrimitiveArray::from_iter(vec![5i32]).into_array();
        assert_arrays_eq!(taken, expected);
    }

    #[test]
    fn ree_take_sorted_boundaries() {
        let taken = ree_array()
            .take(buffer![0, 2, 3, 6, 8, 11].into_array())
            .unwrap();
        let expected = PrimitiveArray::from_iter(vec![1i32, 1, 4, 2, 5, 5]).into_array();
        assert_arrays_eq!(taken, expected);
    }

    #[test]
    #[should_panic]
    fn ree_take_out_of_bounds() {
        let _array = ree_array()
            .take(buffer![12].into_array())
            .unwrap()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
    }

    #[test]
    fn sliced_take() {
        let sliced = ree_array().slice(4..9).unwrap();
        let taken = sliced.take(buffer![1, 3, 4].into_array()).unwrap();

        let expected = PrimitiveArray::from_iter(vec![4i32, 2, 5]).into_array();
        assert_arrays_eq!(taken, expected);
    }

    #[test]
    fn ree_take_nullable() {
        let taken = ree_array()
            .take(PrimitiveArray::from_option_iter([Some(1), None]).into_array())
            .unwrap();

        let expected = PrimitiveArray::from_option_iter([Some(1i32), None]);
        assert_arrays_eq!(taken, expected.into_array());
    }

    #[test]
    fn ree_take_null_index_skips_out_of_bounds_value() {
        let indices = PrimitiveArray::new(
            buffer![1u64, 12],
            Validity::Array(BoolArray::from_iter([true, false]).into_array()),
        );
        let taken = ree_array().take(indices.into_array()).unwrap();

        let expected = PrimitiveArray::from_option_iter([Some(1i32), None]);
        assert_arrays_eq!(taken, expected.into_array());
    }

    #[rstest]
    #[case(ree_array())]
    #[case(RunEnd::encode(
        buffer![1u8, 1, 2, 2, 2, 3, 3, 3, 3, 4].into_array(),
        &mut LEGACY_SESSION.create_execution_ctx(),
    ).unwrap())]
    #[case(RunEnd::encode(
        PrimitiveArray::from_option_iter([
            Some(10),
            Some(10),
            None,
            None,
            Some(20),
            Some(20),
            Some(20),
        ])
        .into_array(),
        &mut LEGACY_SESSION.create_execution_ctx(),
    ).unwrap())]
    #[case(RunEnd::encode(buffer![42i32, 42, 42, 42, 42].into_array(),
        &mut LEGACY_SESSION.create_execution_ctx())
        .unwrap())]
    #[case(RunEnd::encode(
        buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array(),
        &mut LEGACY_SESSION.create_execution_ctx(),
    ).unwrap())]
    #[case({
        let mut values = Vec::new();
        for i in 0..20 {
            for _ in 0..=i {
                values.push(i);
            }
        }
        RunEnd::encode(
            PrimitiveArray::from_iter(values).into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap()
    })]
    fn test_take_runend_conformance(#[case] array: RunEndArray) {
        test_take_conformance(&array.into_array());
    }

    #[rstest]
    #[case(ree_array().slice(3..6).unwrap())]
    #[case({
        let array = RunEnd::encode(
            buffer![1i32, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3].into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        array.slice(2..8).unwrap()
    })]
    fn test_take_sliced_runend_conformance(#[case] sliced: ArrayRef) {
        test_take_conformance(&sliced);
    }
}
