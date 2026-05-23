#![cfg(all(feature = "simd", target_arch = "aarch64"))]
#![allow(unsafe_code)]

use crate::error::FieldError;
use crate::simd::scalar;
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
unsafe fn apply_sub_div_neon_impl(
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
    let lanes = 4_usize;
    let aligned_len = values.len() / lanes * lanes;

    let sub_v = vdupq_n_f32(sub);
    let div_v = vdupq_n_f32(div);

    let mut i = 0_usize;
    while i < aligned_len {
        // SAFETY: `i + lanes <= aligned_len <= values.len()` and load/store use
        // bounded pointers derived from valid input/output slices.
        let input_v = unsafe { vld1q_f32(values.as_ptr().add(i)) };
        let diff_v = vsubq_f32(input_v, sub_v);
        let out_v = vdivq_f32(diff_v, div_v);
        // SAFETY: same bounds argument as the load above for output slice.
        unsafe { vst1q_f32(out.as_mut_ptr().add(i), out_v) };
        i += lanes;
    }

    if aligned_len < values.len() {
        let tail = scalar::apply_sub_div_scalar(&values[aligned_len..], sub, div)?;
        out[aligned_len..].copy_from_slice(&tail);
    }

    Ok(out)
}

pub unsafe fn apply_sub_div_neon(
    values: &[f32],
    sub: f32,
    div: f32,
) -> Result<Vec<f32>, FieldError> {
    // SAFETY: caller guarantees NEON availability before calling this wrapper.
    unsafe { apply_sub_div_neon_impl(values, sub, div) }
}
