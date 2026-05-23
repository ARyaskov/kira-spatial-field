use crate::{
    error::FieldError, metadata::FieldMetadata, normalization::NormalizationFlags,
    reduction::ReductionMethod, simd,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Computes a deterministic creation hash for provenance and reproducibility.
fn compute_creation_hash(
    domain_id: u64,
    source_genes: &[String],
    reduction_method: &ReductionMethod,
    normalization_flags: &NormalizationFlags,
    axis_hash: Option<[u8; 32]>,
    values: &[f32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();

    hasher.update(domain_id.to_le_bytes());

    for gene_id in source_genes {
        hasher.update(gene_id.as_bytes());
        hasher.update([b'\n']);
    }

    hasher.update([reduction_method.discriminant()]);
    if let Some(payload) = reduction_method.hash_payload() {
        hasher.update(payload);
    }

    hasher.update([normalization_flags.log1p as u8]);
    hasher.update([normalization_flags.zscore_global as u8]);
    hasher.update([normalization_flags.zscore_masked as u8]);
    hasher.update([normalization_flags.minmax_scale as u8]);

    if let Some(axis_hash_bytes) = axis_hash {
        hasher.update(axis_hash_bytes);
    }

    #[cfg(target_endian = "little")]
    {
        let byte_view: &[u8] = bytemuck::cast_slice(values);
        hasher.update(byte_view);
    }
    #[cfg(target_endian = "big")]
    {
        for value in values {
            hasher.update(value.to_le_bytes());
        }
    }

    let digest = hasher.finalize();
    let mut out = [0_u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn validate_invariants(values: &[f32]) -> Result<(), FieldError> {
    if values.is_empty() {
        return Err(FieldError::InvalidValues);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(FieldError::InvalidValues);
    }
    Ok(())
}

/// Deterministic scalar field container for spatial operators.
///
/// Immutable after construction. Arc-backed values allow shared ownership.
/// Every transform returns a new `Field`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    domain_id: u64,
    values: Arc<[f32]>,
    metadata: FieldMetadata,
}

impl Field {
    pub(crate) fn from_parts(domain_id: u64, values: Vec<f32>, metadata: FieldMetadata) -> Self {
        Self {
            domain_id,
            values: values.into(),
            metadata,
        }
    }

    pub(crate) fn with_metadata(mut self, metadata: FieldMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub(crate) fn finalize_creation_hash(mut self) -> Result<Self, FieldError> {
        validate_invariants(self.values.as_ref())?;

        let hash = compute_creation_hash(
            self.domain_id,
            self.metadata.source_genes(),
            self.metadata.reduction_method(),
            self.metadata.normalization_flags(),
            self.metadata.axis_hash(),
            self.values.as_ref(),
        );
        if hash.iter().all(|byte| *byte == 0) {
            return Err(FieldError::InvalidMetadata);
        }

        self.metadata.set_creation_hash(hash);
        Ok(self)
    }

    /// Build a `Field` directly from pre-computed values, bypassing the
    /// gene/panel/axis builders. `creation_hash` is computed during this call.
    pub fn from_values(
        domain_id: u64,
        values: Vec<f32>,
        metadata: FieldMetadata,
    ) -> Result<Self, FieldError> {
        Self::from_parts(domain_id, values, metadata).finalize_creation_hash()
    }

    /// Apply a deterministic element-wise transform left-to-right.
    pub fn apply<F>(&self, field_name_suffix: &str, mut f: F) -> Result<Self, FieldError>
    where
        F: FnMut(f32) -> f32,
    {
        let mut out = Vec::with_capacity(self.values.len());
        for &v in self.values.iter() {
            let transformed = f(v);
            if !transformed.is_finite() {
                return Err(FieldError::InvalidValues);
            }
            out.push(transformed);
        }
        let mut metadata = self.metadata.clone();
        metadata.append_field_name_suffix(field_name_suffix);
        Self::from_parts(self.domain_id, out, metadata).finalize_creation_hash()
    }

    /// Returns immutable field values in domain bin order.
    pub fn values(&self) -> &[f32] {
        self.values.as_ref()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the bound spatial domain identity.
    pub fn domain_id(&self) -> u64 {
        self.domain_id
    }

    /// Returns field metadata captured at creation.
    pub fn metadata(&self) -> &FieldMetadata {
        &self.metadata
    }

    /// Deterministic element-wise `ln(1 + x)`.
    pub fn log1p(&self) -> Result<Self, FieldError> {
        let out = simd::apply_log1p(self.values.as_ref())?;
        if out.len() != self.values.len() {
            return Err(FieldError::InvalidValues);
        }
        validate_invariants(&out)?;

        let mut metadata = self.metadata.clone();
        metadata.append_field_name_suffix("::log1p");
        let mut flags = *metadata.normalization_flags();
        flags.log1p = true;
        metadata.set_normalization_flags(flags);
        Self::from_parts(self.domain_id, out, metadata).finalize_creation_hash()
    }

    /// Deterministic global z-score normalization over all bins.
    pub fn zscore_global(&self) -> Result<Self, FieldError> {
        let n = self.values.len();
        if n == 0 {
            return Err(FieldError::InvalidValues);
        }
        validate_invariants(self.values.as_ref())?;

        // Welford in f64 to avoid catastrophic cancellation at >10^6 elements.
        let mut count: u64 = 0;
        let mut mean64 = 0.0_f64;
        let mut m2 = 0.0_f64;
        for &value in self.values.iter() {
            let x = value as f64;
            count += 1;
            let delta = x - mean64;
            mean64 += delta / count as f64;
            let delta2 = x - mean64;
            m2 += delta * delta2;
        }
        let variance64 = m2 / count as f64;
        if !mean64.is_finite() || !variance64.is_finite() {
            return Err(FieldError::InvalidValues);
        }
        let sigma64 = variance64.sqrt();
        if !sigma64.is_finite() {
            return Err(FieldError::InvalidValues);
        }
        let mean = mean64 as f32;
        let sigma = sigma64 as f32;

        let mut out = Vec::with_capacity(n);
        if sigma == 0.0 {
            out.resize(n, 0.0);
        } else {
            out = simd::apply_sub_div(self.values.as_ref(), mean, sigma)?;
        }
        if out.len() != n {
            return Err(FieldError::InvalidValues);
        }
        validate_invariants(&out)?;

        let mut metadata = self.metadata.clone();
        metadata.append_field_name_suffix("::zscore");
        let mut flags = *metadata.normalization_flags();
        flags.zscore_global = true;
        metadata.set_normalization_flags(flags);
        Self::from_parts(self.domain_id, out, metadata).finalize_creation_hash()
    }

    /// Deterministic z-score normalization over masked bins; unmasked bins
    /// are copied unchanged.
    pub fn zscore_masked(&self, mask: &[bool]) -> Result<Self, FieldError> {
        if mask.len() != self.values.len() {
            return Err(FieldError::DomainSizeMismatch);
        }
        validate_invariants(self.values.as_ref())?;

        let mut count: u64 = 0;
        let mut mean64 = 0.0_f64;
        let mut m2 = 0.0_f64;
        for (idx, &is_masked) in mask.iter().enumerate() {
            if is_masked {
                let x = self.values[idx] as f64;
                count += 1;
                let delta = x - mean64;
                mean64 += delta / count as f64;
                let delta2 = x - mean64;
                m2 += delta * delta2;
            }
        }
        if count == 0 {
            return Err(FieldError::InvalidValues);
        }
        let variance64 = m2 / count as f64;
        if !mean64.is_finite() || !variance64.is_finite() {
            return Err(FieldError::InvalidValues);
        }
        let sigma64 = variance64.sqrt();
        if !sigma64.is_finite() {
            return Err(FieldError::InvalidValues);
        }
        let mean = mean64 as f32;
        let sigma = sigma64 as f32;

        let mut out = Vec::with_capacity(self.values.len());
        for (idx, &value) in self.values.iter().enumerate() {
            if mask[idx] {
                if sigma == 0.0 {
                    out.push(0.0);
                } else {
                    let transformed = (value - mean) / sigma;
                    if !transformed.is_finite() {
                        return Err(FieldError::InvalidValues);
                    }
                    out.push(transformed);
                }
            } else {
                out.push(value);
            }
        }

        let mut metadata = self.metadata.clone();
        metadata.append_field_name_suffix("::zscore_masked");
        let mut flags = *metadata.normalization_flags();
        flags.zscore_masked = true;
        metadata.set_normalization_flags(flags);
        Self::from_parts(self.domain_id, out, metadata).finalize_creation_hash()
    }

    /// Deterministic global min-max scaling.
    pub fn minmax_scale(&self) -> Result<Self, FieldError> {
        let n = self.values.len();
        if n == 0 {
            return Err(FieldError::InvalidValues);
        }
        validate_invariants(self.values.as_ref())?;

        let mut min_value = self.values[0];
        let mut max_value = self.values[0];
        for &value in self.values.iter() {
            if value < min_value {
                min_value = value;
            }
            if value > max_value {
                max_value = value;
            }
        }

        let mut out = Vec::with_capacity(n);
        if max_value == min_value {
            out.resize(n, 0.0);
        } else {
            let denom = max_value - min_value;
            if !denom.is_finite() || denom == 0.0 {
                return Err(FieldError::InvalidValues);
            }
            out = simd::apply_sub_div(self.values.as_ref(), min_value, denom)?;
        }
        if out.len() != n {
            return Err(FieldError::InvalidValues);
        }
        validate_invariants(&out)?;

        let mut metadata = self.metadata.clone();
        metadata.append_field_name_suffix("::minmax");
        let mut flags = *metadata.normalization_flags();
        flags.minmax_scale = true;
        metadata.set_normalization_flags(flags);
        Self::from_parts(self.domain_id, out, metadata).finalize_creation_hash()
    }

    /// `log1p` over masked bins; unmasked bins copied unchanged.
    pub fn log1p_masked(&self, mask: &[bool]) -> Result<Self, FieldError> {
        if mask.len() != self.values.len() {
            return Err(FieldError::DomainSizeMismatch);
        }
        validate_invariants(self.values.as_ref())?;

        let mut out = Vec::with_capacity(self.values.len());
        for (idx, &value) in self.values.iter().enumerate() {
            if mask[idx] {
                if !value.is_finite() || value < -1.0 {
                    return Err(FieldError::InvalidValues);
                }
                let t = (1.0_f32 + value).ln();
                if !t.is_finite() {
                    return Err(FieldError::InvalidValues);
                }
                out.push(t);
            } else {
                out.push(value);
            }
        }

        let mut metadata = self.metadata.clone();
        metadata.append_field_name_suffix("::log1p_masked");
        let mut flags = *metadata.normalization_flags();
        flags.log1p = true;
        metadata.set_normalization_flags(flags);
        Self::from_parts(self.domain_id, out, metadata).finalize_creation_hash()
    }

    /// Min-max scaling over masked bins; unmasked bins copied unchanged.
    pub fn minmax_scale_masked(&self, mask: &[bool]) -> Result<Self, FieldError> {
        if mask.len() != self.values.len() {
            return Err(FieldError::DomainSizeMismatch);
        }
        validate_invariants(self.values.as_ref())?;

        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut any = false;
        for (idx, &value) in self.values.iter().enumerate() {
            if mask[idx] {
                any = true;
                if value < min_v {
                    min_v = value;
                }
                if value > max_v {
                    max_v = value;
                }
            }
        }
        if !any {
            return Err(FieldError::InvalidValues);
        }

        let mut out = Vec::with_capacity(self.values.len());
        let denom = max_v - min_v;
        for (idx, &value) in self.values.iter().enumerate() {
            if mask[idx] {
                if denom == 0.0 {
                    out.push(0.0);
                } else {
                    let t = (value - min_v) / denom;
                    if !t.is_finite() {
                        return Err(FieldError::InvalidValues);
                    }
                    out.push(t);
                }
            } else {
                out.push(value);
            }
        }

        let mut metadata = self.metadata.clone();
        metadata.append_field_name_suffix("::minmax_masked");
        let mut flags = *metadata.normalization_flags();
        flags.minmax_scale = true;
        metadata.set_normalization_flags(flags);
        Self::from_parts(self.domain_id, out, metadata).finalize_creation_hash()
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Field) -> Result<Self, FieldError> {
        self.binary_op(other, "::add", |a, b| a + b)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Field) -> Result<Self, FieldError> {
        self.binary_op(other, "::sub", |a, b| a - b)
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Field) -> Result<Self, FieldError> {
        self.binary_op(other, "::mul", |a, b| a * b)
    }

    /// Element-wise division.
    pub fn div(&self, other: &Field) -> Result<Self, FieldError> {
        self.binary_op(other, "::div", |a, b| a / b)
    }

    /// Rank-normalize values into uniform `[0, 1]` order statistics.
    /// Ties are broken deterministically by bin index.
    pub fn rank_normalize(&self) -> Result<Self, FieldError> {
        let n = self.values.len();
        if n == 0 {
            return Err(FieldError::InvalidValues);
        }
        validate_invariants(self.values.as_ref())?;

        let mut indices: Vec<u32> = (0..n as u32).collect();
        indices.sort_by(|&a, &b| {
            self.values[a as usize]
                .total_cmp(&self.values[b as usize])
                .then(a.cmp(&b))
        });

        let mut out = vec![0.0_f32; n];
        let denom = if n > 1 { (n - 1) as f32 } else { 1.0 };
        for (rank, &idx) in indices.iter().enumerate() {
            out[idx as usize] = rank as f32 / denom;
        }

        let mut metadata = self.metadata.clone();
        metadata.append_field_name_suffix("::rank");
        Self::from_parts(self.domain_id, out, metadata).finalize_creation_hash()
    }

    fn binary_op<F>(&self, other: &Field, suffix: &str, mut op: F) -> Result<Self, FieldError>
    where
        F: FnMut(f32, f32) -> f32,
    {
        if self.domain_id != other.domain_id {
            return Err(FieldError::DomainSizeMismatch);
        }
        if self.values.len() != other.values.len() {
            return Err(FieldError::DomainSizeMismatch);
        }
        let mut out = Vec::with_capacity(self.values.len());
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            let r = op(*a, *b);
            if !r.is_finite() {
                return Err(FieldError::InvalidValues);
            }
            out.push(r);
        }
        let mut metadata = self.metadata.clone();
        metadata.append_field_name_suffix(suffix);
        Self::from_parts(self.domain_id, out, metadata).finalize_creation_hash()
    }
}

impl PartialEq for Field {
    fn eq(&self, other: &Self) -> bool {
        match (self.metadata.creation_hash(), other.metadata.creation_hash()) {
            (Some(a), Some(b)) => a == b,
            _ => self.domain_id == other.domain_id && self.values.as_ref() == other.values.as_ref(),
        }
    }
}

impl Eq for Field {}

impl std::fmt::Display for Field {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let hash_prefix = self
            .metadata
            .creation_hash()
            .map(|h| {
                let mut s = String::with_capacity(16);
                for b in &h[..8] {
                    s.push_str(&format!("{:02x}", b));
                }
                s
            })
            .unwrap_or_else(|| "<no-hash>".to_string());
        write!(
            f,
            "Field {{ name={}, domain={}, len={}, hash={} }}",
            self.metadata.field_name(),
            self.domain_id,
            self.values.len(),
            hash_prefix,
        )
    }
}
