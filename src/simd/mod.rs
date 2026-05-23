#![allow(unsafe_code)]

use crate::error::FieldError;

pub mod scalar;

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx2;
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub mod neon;

// `log1p` is scalar-only to preserve bitwise equality across SIMD toggles
// (SIMD `ln` approximations differ from `f32::ln` by 1–4 ulp).
pub fn apply_log1p(values: &[f32]) -> Result<Vec<f32>, FieldError> {
    scalar::apply_log1p_scalar(values)
}

pub fn apply_sub_div(values: &[f32], sub: f32, div: f32) -> Result<Vec<f32>, FieldError> {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: Runtime feature detection guarantees AVX2 support.
            return unsafe { avx2::apply_sub_div_avx2(values, sub, div) };
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: Runtime feature detection guarantees NEON support.
            return unsafe { neon::apply_sub_div_neon(values, sub, div) };
        }
    }

    scalar::apply_sub_div_scalar(values, sub, div)
}
