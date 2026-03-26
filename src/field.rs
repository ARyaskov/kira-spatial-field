use crate::{
    error::FieldError, metadata::FieldMetadata, normalization::NormalizationFlags,
    reduction::ReductionMethod, simd,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Computes a deterministic creation hash for provenance and reproducibility.
///
/// The hash is platform-independent because bytes are fed manually in a fixed
/// order with explicit little-endian encoding for numeric values.
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

    for value in values {
        hasher.update(value.to_le_bytes());
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
/// `Field` is logically immutable after construction.
/// Value storage is Arc-backed to prevent accidental mutation while allowing
/// efficient shared ownership in future cache-oriented integrations.
/// Every transform returns a new `Field` with preserved determinism.
///
/// Stage 0 invariants are documented here and enforced in later stages:
/// 1. The field contains exactly one scalar value per spatial bin.
/// 2. Value ordering must exactly match the owning SpatialDomain ordering.
/// 3. Values must not contain NaN or infinite numbers.
/// 4. No implicit normalization is allowed at any API boundary.
/// 5. `domain_id` binds the field identity to one concrete SpatialDomain.
/// 6. Floating-point operations used to produce this field must be deterministic.
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
        // `creation_hash` provides bitwise reproducibility: any change in
        // construction inputs changes the digest, while identical inputs
        // produce identical hashes suitable for provenance and cache keys.
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

    /// Returns immutable field values in domain bin order.
    pub fn values(&self) -> &[f32] {
        self.values.as_ref()
    }

    /// Returns the bound spatial domain identity.
    pub fn domain_id(&self) -> u64 {
        self.domain_id
    }

    /// Returns field metadata captured at creation.
    pub fn metadata(&self) -> &FieldMetadata {
        &self.metadata
    }

    /// Applies deterministic element-wise `ln(1 + x)` normalization.
    ///
    /// Determinism notes:
    /// - Sequential left-to-right iteration over bins.
    /// - Fixed floating-point operation order per element.
    /// - No implicit scaling and no hidden state.
    /// - Identical input yields identical output.
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

    /// Applies deterministic global z-score normalization over all bins.
    ///
    /// Determinism notes:
    /// - Sequential two-pass computation (mean, then variance).
    /// - Fixed floating-point accumulation order.
    /// - No implicit scaling and no hidden state.
    /// - Identical input yields identical output.
    pub fn zscore_global(&self) -> Result<Self, FieldError> {
        let n = self.values.len();
        if n == 0 {
            return Err(FieldError::InvalidValues);
        }
        validate_invariants(self.values.as_ref())?;

        let mut sum = 0.0_f32;
        for &value in self.values.iter() {
            sum += value;
        }
        let mean = sum / n as f32;
        if !mean.is_finite() {
            return Err(FieldError::InvalidValues);
        }

        let mut sq_sum = 0.0_f32;
        for &value in self.values.iter() {
            let diff = value - mean;
            sq_sum += diff * diff;
        }
        let variance = sq_sum / n as f32;
        if !variance.is_finite() {
            return Err(FieldError::InvalidValues);
        }
        let sigma = variance.sqrt();
        if !sigma.is_finite() {
            return Err(FieldError::InvalidValues);
        }

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

    /// Applies deterministic masked z-score normalization.
    ///
    /// Determinism notes:
    /// - Sequential two-pass computation over masked bins only.
    /// - Fixed floating-point accumulation order.
    /// - Unmasked bins are copied unchanged.
    /// - No implicit scaling and no hidden state.
    /// - Identical input yields identical output.
    pub fn zscore_masked(&self, mask: &[bool]) -> Result<Self, FieldError> {
        if mask.len() != self.values.len() {
            return Err(FieldError::DomainSizeMismatch);
        }
        validate_invariants(self.values.as_ref())?;

        let mut count = 0_usize;
        let mut sum = 0.0_f32;
        for (idx, &is_masked) in mask.iter().enumerate() {
            if is_masked {
                count += 1;
                sum += self.values[idx];
            }
        }
        if count == 0 {
            return Err(FieldError::InvalidValues);
        }
        let mean = sum / count as f32;
        if !mean.is_finite() {
            return Err(FieldError::InvalidValues);
        }

        let mut sq_sum = 0.0_f32;
        for (idx, &is_masked) in mask.iter().enumerate() {
            if is_masked {
                let diff = self.values[idx] - mean;
                sq_sum += diff * diff;
            }
        }
        let variance = sq_sum / count as f32;
        if !variance.is_finite() {
            return Err(FieldError::InvalidValues);
        }
        let sigma = variance.sqrt();
        if !sigma.is_finite() {
            return Err(FieldError::InvalidValues);
        }

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

    /// Applies deterministic global min-max scaling.
    ///
    /// Determinism notes:
    /// - Sequential min/max scan followed by sequential scaling.
    /// - Fixed floating-point operation order.
    /// - No implicit scaling and no hidden state.
    /// - Identical input yields identical output.
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
}
