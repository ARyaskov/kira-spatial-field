use crate::error::FieldError;

pub fn apply_log1p_scalar(values: &[f32]) -> Result<Vec<f32>, FieldError> {
    let mut out = Vec::with_capacity(values.len());
    for &value in values {
        if !value.is_finite() || value < -1.0 {
            return Err(FieldError::InvalidValues);
        }
        let transformed = (1.0_f32 + value).ln();
        if !transformed.is_finite() {
            return Err(FieldError::InvalidValues);
        }
        out.push(transformed);
    }
    Ok(out)
}

pub fn apply_sub_div_scalar(values: &[f32], sub: f32, div: f32) -> Result<Vec<f32>, FieldError> {
    if !sub.is_finite() || !div.is_finite() || div == 0.0 {
        return Err(FieldError::InvalidValues);
    }

    let mut out = Vec::with_capacity(values.len());
    for &value in values {
        if !value.is_finite() {
            return Err(FieldError::InvalidValues);
        }
        let transformed = (value - sub) / div;
        if !transformed.is_finite() {
            return Err(FieldError::InvalidValues);
        }
        out.push(transformed);
    }
    Ok(out)
}
