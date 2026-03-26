//! Immutable biological axis definition used for reproducible field workflows.
//!
//! `AxisDefinition` is immutable after validated construction.
//! The stored SHA-256 hash guarantees reproducibility for the exact definition.
//! Any semantic change to an axis requires a version bump.

use crate::{error::FieldError, normalization::NormalizationFlags, reduction::ReductionMethod};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisDefinition {
    axis_id: String,
    version: u32,
    gene_ids: Vec<String>,
    weights: Option<Vec<f32>>,
    reduction_method: ReductionMethod,
    default_normalization: NormalizationFlags,
    definition_hash: [u8; 32],
}

impl AxisDefinition {
    pub fn new(
        axis_id: String,
        version: u32,
        mut gene_ids: Vec<String>,
        weights: Option<Vec<f32>>,
        reduction_method: ReductionMethod,
        default_normalization: NormalizationFlags,
    ) -> Result<Self, FieldError> {
        let sorted_weights = match weights {
            Some(w) => {
                if w.len() != gene_ids.len() {
                    return Err(FieldError::InvalidReduction);
                }

                let mut pairs: Vec<(String, f32)> = gene_ids.into_iter().zip(w).collect();
                pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                gene_ids = pairs.iter().map(|(gene_id, _)| gene_id.clone()).collect();
                Some(pairs.into_iter().map(|(_, weight)| weight).collect())
            }
            None => {
                gene_ids.sort_unstable();
                None
            }
        };

        if gene_ids.windows(2).any(|window| window[0] == window[1]) {
            return Err(FieldError::InvalidMetadata);
        }

        match (&reduction_method, sorted_weights.as_ref()) {
            (ReductionMethod::Weighted, None) => return Err(FieldError::InvalidReduction),
            (ReductionMethod::Weighted, Some(_)) => {}
            (_, Some(_)) => return Err(FieldError::InvalidReduction),
            (_, None) => {}
        }

        let definition_hash = Self::compute_hash(
            &axis_id,
            version,
            &gene_ids,
            sorted_weights.as_deref(),
            &reduction_method,
            &default_normalization,
        );

        Ok(Self {
            axis_id,
            version,
            gene_ids,
            weights: sorted_weights,
            reduction_method,
            default_normalization,
            definition_hash,
        })
    }

    fn compute_hash(
        axis_id: &str,
        version: u32,
        gene_ids: &[String],
        weights: Option<&[f32]>,
        reduction_method: &ReductionMethod,
        default_normalization: &NormalizationFlags,
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();

        hasher.update(axis_id.as_bytes());
        hasher.update(version.to_le_bytes());

        let gene_bytes = gene_ids.join("\n");
        hasher.update(gene_bytes.as_bytes());

        if let Some(weight_values) = weights {
            for weight in weight_values {
                hasher.update(weight.to_le_bytes());
            }
        }

        hasher.update([reduction_method.discriminant()]);
        if let Some(payload) = reduction_method.hash_payload() {
            hasher.update(payload);
        }

        hasher.update([default_normalization.log1p as u8]);
        hasher.update([default_normalization.zscore_global as u8]);
        hasher.update([default_normalization.zscore_masked as u8]);
        hasher.update([default_normalization.minmax_scale as u8]);

        let digest = hasher.finalize();
        let mut out = [0_u8; 32];
        out.copy_from_slice(&digest);
        out
    }

    pub fn axis_id(&self) -> &str {
        &self.axis_id
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn gene_ids(&self) -> &[String] {
        &self.gene_ids
    }

    pub fn weights(&self) -> Option<&[f32]> {
        self.weights.as_deref()
    }

    pub fn reduction_method(&self) -> &ReductionMethod {
        &self.reduction_method
    }

    pub fn default_normalization(&self) -> &NormalizationFlags {
        &self.default_normalization
    }

    pub fn definition_hash(&self) -> &[u8; 32] {
        &self.definition_hash
    }
}
