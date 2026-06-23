// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::ToI256;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Decimal {
    fn cast(array: ArrayView<'_, Decimal>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Only nullability changes within the same decimal dtype are reducible without execution.
        // Precision/scale changes need the kernel.
        let DType::Decimal(to_decimal_dtype, to_nullability) = dtype else {
            return Ok(None);
        };
        let DType::Decimal(from_decimal_dtype, _) = array.dtype() else {
            vortex_panic!(
                "DecimalArray must have decimal dtype, got {:?}",
                array.dtype()
            );
        };

        if from_decimal_dtype != to_decimal_dtype {
            return Ok(None);
        }

        let Some(new_validity) = array
            .validity()?
            .trivially_cast_nullability(*to_nullability, array.len())?
        else {
            return Ok(None);
        };

        // SAFETY: validity has the same length, only its nullability tag changes.
        unsafe {
            Ok(Some(
                DecimalArray::new_unchecked_handle(
                    array.buffer_handle().clone(),
                    array.values_type(),
                    *to_decimal_dtype,
                    new_validity,
                )
                .into_array(),
            ))
        }
    }
}

impl CastKernel for Decimal {
    fn cast(
        array: ArrayView<'_, Decimal>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Early return if not casting to decimal
        let DType::Decimal(to_decimal_dtype, to_nullability) = dtype else {
            return Ok(None);
        };
        let DType::Decimal(from_decimal_dtype, _) = array.dtype() else {
            vortex_panic!(
                "DecimalArray must have decimal dtype, got {:?}",
                array.dtype()
            );
        };

        // If the dtype is exactly the same, return self
        if array.dtype() == dtype {
            return Ok(Some(array.array().clone()));
        }

        // Cast the validity to the new nullability (shared by every path below).
        let new_validity = array
            .validity()?
            .cast_nullability(*to_nullability, array.len(), ctx)?;

        // A scale change rescales every mantissa by 10^(to_scale - from_scale).
        // DataFusion routinely coerces a decimal comparison to a higher scale
        // (its float-literal coercion lands on e.g. `decimal(53, 15)`), and the
        // pruning predicate then casts the column's scale-2 stat up to scale 15
        // — so a scan that prunes on a decimal column depends on this working.
        // The result is computed in i256 and stored as i256, a compatible
        // (wide-enough) physical type for any target precision.
        if from_decimal_dtype.scale() != to_decimal_dtype.scale() {
            let rescaled = rescale_decimal_buffer(array, to_decimal_dtype.scale())?;
            return Ok(Some(
                DecimalArray::new(rescaled, *to_decimal_dtype, new_validity).into_array(),
            ));
        }

        // Same scale: downcasting precision is not yet supported.
        if to_decimal_dtype.precision() < from_decimal_dtype.precision() {
            vortex_bail!(
                "Downcasting decimal from precision {} to {} not yet implemented",
                from_decimal_dtype.precision(),
                to_decimal_dtype.precision()
            );
        }

        // Same scale, wider-or-equal precision: widen the physical type if needed.
        let target_values_type = DecimalType::smallest_decimal_value_type(to_decimal_dtype);
        let array = if target_values_type > array.values_type() {
            upcast_decimal_values(array, target_values_type)?
        } else {
            array.array().as_::<Decimal>().into_owned()
        };

        // SAFETY: new_validity same length as previous validity, just cast
        unsafe {
            Ok(Some(
                DecimalArray::new_unchecked_handle(
                    array.buffer_handle().clone(),
                    array.values_type(),
                    *to_decimal_dtype,
                    new_validity,
                )
                .into_array(),
            ))
        }
    }
}

/// Rescale every mantissa of `array` from its current scale to `to_scale`, returning the
/// rescaled values as an `i256` buffer (a compatible physical type for any target precision,
/// so the caller does not need to pick the narrowest type).
///
/// Widening the scale multiplies by `10^delta` and is exact. Narrowing divides by `10^|delta|`
/// with round-half-away-from-zero, matching Arrow's decimal-cast rounding. A value that no
/// longer fits `i256`, or a `10^|delta|` factor beyond `i256`, surfaces as an error rather than
/// silently wrapping.
fn rescale_decimal_buffer(
    array: ArrayView<'_, Decimal>,
    to_scale: i8,
) -> VortexResult<Buffer<i256>> {
    let from_scale = array.decimal_dtype().scale();
    match_each_decimal_value_type!(array.values_type(), |F| {
        array
            .buffer::<F>()
            .iter()
            .map(|&v| {
                let v = v
                    .to_i256()
                    .vortex_expect("a native decimal value always widens to i256");
                DecimalValue::I256(v)
                    .rescale(from_scale, to_scale)
                    .map(|rescaled| rescaled.as_i256())
                    .ok_or_else(|| {
                        vortex_err!(
                            "decimal value overflows i256 when rescaling to scale {to_scale}"
                        )
                    })
            })
            .collect::<VortexResult<Buffer<i256>>>()
    })
}

/// Upcast a DecimalArray to a wider physical representation (e.g., i32 -> i64) while keeping
/// the same precision and scale.
///
/// This is useful when you need to widen the underlying storage type to accommodate operations
/// that might overflow the current representation, or to match the physical type expected by
/// downstream consumers.
///
/// # Errors
///
/// Returns an error if `to_values_type` is narrower than the array's current values type.
/// Only upcasting (widening) is supported.
pub fn upcast_decimal_values(
    array: ArrayView<'_, Decimal>,
    to_values_type: DecimalType,
) -> VortexResult<DecimalArray> {
    let from_values_type = array.values_type();

    // If already the target type, just clone
    if from_values_type == to_values_type {
        return Ok(array.array().as_::<Decimal>().into_owned());
    }

    // Only allow upcasting (widening)
    if to_values_type < from_values_type {
        vortex_bail!(
            "Cannot downcast decimal values from {:?} to {:?}. Only upcasting is supported.",
            from_values_type,
            to_values_type
        );
    }

    let decimal_dtype = array.decimal_dtype();
    let validity = array.validity()?;

    // Use match_each_decimal_value_type to dispatch based on source and target types
    match_each_decimal_value_type!(from_values_type, |F| {
        let from_buffer = array.buffer::<F>();
        match_each_decimal_value_type!(to_values_type, |T| {
            let to_buffer = upcast_decimal_buffer::<F, T>(from_buffer);
            Ok(DecimalArray::new(to_buffer, decimal_dtype, validity))
        })
    })
}

/// Upcast a buffer of decimal values from type F to type T.
/// Since T is wider than F, this conversion never fails.
fn upcast_decimal_buffer<F: NativeDecimalType, T: NativeDecimalType>(from: Buffer<F>) -> Buffer<T> {
    from.iter()
        .map(|&v| T::from(v).vortex_expect("upcast should never fail"))
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use super::upcast_decimal_values;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::RecursiveCanonical;
    use crate::VortexSessionExecute;
    use crate::arrays::DecimalArray;
    use crate::builtins::ArrayBuiltins;
    #[expect(deprecated)]
    use crate::canonical::ToCanonical as _;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::DecimalType;
    use crate::dtype::Nullability;
    use crate::dtype::i256;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    /// Cast through the execution path (the deprecated lazy `.cast()` alone only
    /// runs the reduce, which relabels the dtype without rescaling — so a value
    /// assertion must force the kernel via `execute`).
    fn cast_and_execute(array: DecimalArray, to: DType) -> ArrayRef {
        #[expect(deprecated)]
        array
            .into_array()
            .cast(to)
            .unwrap()
            .execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .0
            .into_array()
    }

    #[test]
    fn cast_decimal_widens_scale_exactly() {
        // 123.45, 678.90 at (10, 2) cast to (20, 4): each mantissa multiplies by 100.
        let array = DecimalArray::new(
            buffer![12345i32, 67890],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );
        let to_dtype = DecimalDType::new(20, 4);
        let to = DType::Decimal(to_dtype, Nullability::NonNullable);
        let casted = cast_and_execute(array, to.clone());
        assert_eq!(casted.dtype(), &to);
        let expect = |raw: i128| {
            Scalar::decimal(
                DecimalValue::I256(i256::from_i128(raw)),
                to_dtype,
                Nullability::NonNullable,
            )
        };
        assert_eq!(casted.scalar_at(0).unwrap(), expect(1_234_500));
        assert_eq!(casted.scalar_at(1).unwrap(), expect(6_789_000));
    }

    #[test]
    fn cast_decimal_narrows_scale_rounds_half_away_from_zero() {
        // (10, 4) -> (10, 2): divide each mantissa by 100, rounding half away from zero.
        // 1.2345 -> 1.23 (123.45 -> 123); 1.2355 -> 1.24 (123.55 -> 124); -1.2355 -> -1.24.
        let array = DecimalArray::new(
            buffer![12345i32, 12355, -12355],
            DecimalDType::new(10, 4),
            Validity::NonNullable,
        );
        let to_dtype = DecimalDType::new(10, 2);
        let to = DType::Decimal(to_dtype, Nullability::NonNullable);
        let casted = cast_and_execute(array, to);
        let expect = |raw: i128| {
            Scalar::decimal(
                DecimalValue::I256(i256::from_i128(raw)),
                to_dtype,
                Nullability::NonNullable,
            )
        };
        assert_eq!(casted.scalar_at(0).unwrap(), expect(123));
        assert_eq!(casted.scalar_at(1).unwrap(), expect(124));
        assert_eq!(casted.scalar_at(2).unwrap(), expect(-124));
    }

    #[test]
    fn cast_decimal_to_nullable() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalArray::new(
            buffer![100i32, 200, 300],
            decimal_dtype,
            Validity::NonNullable,
        );

        // Cast to nullable
        let nullable_dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        #[expect(deprecated)]
        let casted = array
            .into_array()
            .cast(nullable_dtype.clone())
            .unwrap()
            .to_decimal();

        assert_eq!(casted.dtype(), &nullable_dtype);
        assert!(matches!(casted.validity(), Ok(Validity::AllValid)));
        assert_eq!(casted.len(), 3);
    }

    #[test]
    fn cast_nullable_to_non_nullable() {
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create nullable array with no nulls
        let array = DecimalArray::new(buffer![100i32, 200, 300], decimal_dtype, Validity::AllValid);

        // Cast to non-nullable
        let non_nullable_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        #[expect(deprecated)]
        let casted = array
            .into_array()
            .cast(non_nullable_dtype.clone())
            .unwrap()
            .to_decimal();

        assert_eq!(casted.dtype(), &non_nullable_dtype);
        assert!(matches!(casted.validity(), Ok(Validity::NonNullable)));
    }

    #[test]
    #[should_panic(expected = "Cannot cast array with invalid values to non-nullable type")]
    fn cast_nullable_with_nulls_to_non_nullable_fails() {
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create nullable array with nulls
        let array = DecimalArray::from_option_iter([Some(100i32), None, Some(300)], decimal_dtype);

        // Attempt to cast to non-nullable should fail
        let non_nullable_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        #[expect(deprecated)]
        let result = array
            .into_array()
            .cast(non_nullable_dtype)
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));
        result.unwrap();
    }

    #[test]
    fn cast_different_scale_rescales() {
        // 1.00 at (10, 2) cast to (15, 3): the mantissa multiplies by 10 -> 1000
        // (= 1.000). Scale changes used to bail; they now rescale.
        let array = DecimalArray::new(
            buffer![100i32],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );
        let to_dtype = DecimalDType::new(15, 3);
        let to = DType::Decimal(to_dtype, Nullability::NonNullable);
        let casted = cast_and_execute(array, to.clone());
        assert_eq!(casted.dtype(), &to);
        assert_eq!(
            casted.scalar_at(0).unwrap(),
            Scalar::decimal(
                DecimalValue::I256(i256::from_i128(1000)),
                to_dtype,
                Nullability::NonNullable,
            ),
        );
    }

    #[test]
    fn cast_downcast_precision_fails() {
        let array = DecimalArray::new(
            buffer![100i64],
            DecimalDType::new(18, 2),
            Validity::NonNullable,
        );

        // Try to downcast precision - not supported
        let smaller_dtype = DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable);
        #[expect(deprecated)]
        let result = array
            .into_array()
            .cast(smaller_dtype)
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Downcasting decimal from precision 18 to 10 not yet implemented")
        );
    }

    #[test]
    fn cast_upcast_precision_succeeds() {
        let array = DecimalArray::new(
            buffer![100i32, 200, 300],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );

        // Cast to higher precision with same scale - should succeed
        let wider_dtype = DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable);
        #[expect(deprecated)]
        let casted = array.into_array().cast(wider_dtype).unwrap().to_decimal();

        assert_eq!(casted.precision(), 38);
        assert_eq!(casted.scale(), 2);
        assert_eq!(casted.len(), 3);
        // Should be stored in i128 now (precision 38 requires i128)
        assert_eq!(casted.values_type(), DecimalType::I128);
    }

    #[test]
    fn cast_to_non_decimal_returns_err() {
        let array = DecimalArray::new(
            buffer![100i32],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );

        // Try to cast to non-decimal type - should fail since no kernel can handle it
        #[expect(deprecated)]
        let result = array
            .into_array()
            .cast(DType::Utf8(Nullability::NonNullable))
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No CastKernel to cast canonical array")
        );
    }

    #[rstest]
    #[case(DecimalArray::new(buffer![100i32, 200, 300], DecimalDType::new(10, 2), Validity::NonNullable))]
    #[case(DecimalArray::new(buffer![10000i64, 20000, 30000], DecimalDType::new(18, 4), Validity::NonNullable))]
    #[case(DecimalArray::from_option_iter([Some(100i32), None, Some(300)], DecimalDType::new(10, 2)))]
    #[case(DecimalArray::new(buffer![42i32], DecimalDType::new(5, 1), Validity::NonNullable))]
    fn test_cast_decimal_conformance(#[case] array: DecimalArray) {
        test_cast_conformance(&array.into_array());
    }

    #[test]
    fn upcast_decimal_values_i32_to_i64() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalArray::new(
            buffer![100i32, 200, 300],
            decimal_dtype,
            Validity::NonNullable,
        );

        assert_eq!(array.values_type(), DecimalType::I32);

        let array = array.as_view();
        let casted = upcast_decimal_values(array, DecimalType::I64).unwrap();

        assert_eq!(casted.values_type(), DecimalType::I64);
        assert_eq!(casted.decimal_dtype(), decimal_dtype);
        assert_eq!(casted.len(), 3);

        // Verify values are preserved
        let buffer = casted.buffer::<i64>();
        assert_eq!(buffer.as_ref(), &[100i64, 200, 300]);
    }

    #[test]
    fn upcast_decimal_values_i64_to_i128() {
        let decimal_dtype = DecimalDType::new(18, 4);
        let array = DecimalArray::new(
            buffer![10000i64, 20000, 30000],
            decimal_dtype,
            Validity::NonNullable,
        );

        let array = array.as_view();
        let casted = upcast_decimal_values(array, DecimalType::I128).unwrap();

        assert_eq!(casted.values_type(), DecimalType::I128);
        assert_eq!(casted.decimal_dtype(), decimal_dtype);

        let buffer = casted.buffer::<i128>();
        assert_eq!(buffer.as_ref(), &[10000i128, 20000, 30000]);
    }

    #[test]
    fn upcast_decimal_values_same_type_returns_clone() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalArray::new(
            buffer![100i32, 200, 300],
            decimal_dtype,
            Validity::NonNullable,
        );

        let array = array.as_view();
        let casted = upcast_decimal_values(array, DecimalType::I32).unwrap();

        assert_eq!(casted.values_type(), DecimalType::I32);
        assert_eq!(casted.decimal_dtype(), decimal_dtype);
    }

    #[test]
    fn upcast_decimal_values_with_nulls() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalArray::from_option_iter([Some(100i32), None, Some(300)], decimal_dtype);

        let array = array.as_view();
        let casted = upcast_decimal_values(array, DecimalType::I64).unwrap();

        assert_eq!(casted.values_type(), DecimalType::I64);
        assert_eq!(casted.len(), 3);

        // Check validity is preserved
        let mask = casted
            .as_ref()
            .validity()
            .unwrap()
            .execute_mask(
                casted.as_ref().len(),
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap();
        assert!(mask.value(0));
        assert!(!mask.value(1));
        assert!(mask.value(2));

        // Check non-null values
        let buffer = casted.buffer::<i64>();
        assert_eq!(buffer[0], 100);
        assert_eq!(buffer[2], 300);
    }

    #[test]
    fn upcast_decimal_values_downcast_fails() {
        let decimal_dtype = DecimalDType::new(18, 4);
        let array = DecimalArray::new(
            buffer![10000i64, 20000, 30000],
            decimal_dtype,
            Validity::NonNullable,
        );

        // Attempt to downcast from i64 to i32 should fail
        let array = array.as_view();
        let result = upcast_decimal_values(array, DecimalType::I32);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot downcast decimal values")
        );
    }
}
