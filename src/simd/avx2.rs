#![cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]

use crate::error::FieldError;
use crate::simd::scalar;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
unsafe fn apply_sub_div_avx2_impl(
    values: &[f32],
    sub: f32,
    div: f32,
) -> Result<Vec<f32>, FieldError> {
    if !sub.is_finite() || !div.is_finite() || div == 0.0 {
        return Err(FieldError::InvalidValues);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(FieldError::InvalidValues);
    }

    let mut out = vec![0.0_f32; values.len()];
    let lanes = 8_usize;
    let aligned_len = values.len() / lanes * lanes;

    let sub_v = _mm256_set1_ps(sub);
    let div_v = _mm256_set1_ps(div);

    let mut i = 0_usize;
    while i < aligned_len {
        // SAFETY: `i + lanes <= aligned_len <= values.len()` and load/store use
        // unaligned intrinsics within bounds of the input/output slices.
        let input_v = unsafe { _mm256_loadu_ps(values.as_ptr().add(i)) };
        let diff_v = _mm256_sub_ps(input_v, sub_v);
        let out_v = _mm256_div_ps(diff_v, div_v);
        // SAFETY: same bounds argument as the load above for output slice.
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i), out_v) };
        i += lanes;
    }

    if aligned_len < values.len() {
        let tail = scalar::apply_sub_div_scalar(&values[aligned_len..], sub, div)?;
        out[aligned_len..].copy_from_slice(&tail);
    }

    if out.iter().any(|value| !value.is_finite()) {
        return Err(FieldError::InvalidValues);
    }

    Ok(out)
}

pub unsafe fn apply_sub_div_avx2(
    values: &[f32],
    sub: f32,
    div: f32,
) -> Result<Vec<f32>, FieldError> {
    // SAFETY: caller guarantees AVX2 availability before calling this wrapper.
    unsafe { apply_sub_div_avx2_impl(values, sub, div) }
}
