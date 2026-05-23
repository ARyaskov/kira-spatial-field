//! Immutable biological axis definition used for reproducible field workflows.

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
    /// Build a validated [`AxisDefinition`]. `gene_ids` are sorted in-place
    /// for canonical, hash-stable ordering.
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

        for (i, gene_id) in gene_ids.iter().enumerate() {
            if i > 0 {
                hasher.update(b"\n");
            }
            hasher.update(gene_id.as_bytes());
        }

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

    /// Start a builder.
    pub fn builder(
        axis_id: impl Into<String>,
        version: u32,
        gene_ids: Vec<String>,
        reduction_method: ReductionMethod,
    ) -> AxisDefinitionBuilder {
        AxisDefinitionBuilder {
            axis_id: axis_id.into(),
            version,
            gene_ids,
            weights: None,
            reduction_method,
            default_normalization: NormalizationFlags::default(),
        }
    }

    /// Merge two axes into the alphabetical union of their gene sets.
    /// For `Weighted`, overlapping weights are summed.
    pub fn merge_union(
        &self,
        other: &AxisDefinition,
        composite_id: impl Into<String>,
        composite_version: u32,
    ) -> Result<Self, FieldError> {
        if std::mem::discriminant(&self.reduction_method)
            != std::mem::discriminant(&other.reduction_method)
        {
            return Err(FieldError::InvalidReduction);
        }

        use std::collections::BTreeMap;
        let mut acc: BTreeMap<String, f32> = BTreeMap::new();
        let add = |acc: &mut BTreeMap<String, f32>, ids: &[String], ws: Option<&[f32]>| {
            for (i, gid) in ids.iter().enumerate() {
                let w = ws.map(|w| w[i]).unwrap_or(1.0);
                let entry = acc.entry(gid.clone()).or_insert(0.0);
                *entry += w;
            }
        };
        add(&mut acc, &self.gene_ids, self.weights.as_deref());
        add(&mut acc, &other.gene_ids, other.weights.as_deref());

        let gene_ids: Vec<String> = acc.keys().cloned().collect();
        let weights: Option<Vec<f32>> = match &self.reduction_method {
            ReductionMethod::Weighted => Some(acc.values().copied().collect()),
            _ => None,
        };

        AxisDefinition::new(
            composite_id.into(),
            composite_version,
            gene_ids,
            weights,
            self.reduction_method.clone(),
            self.default_normalization,
        )
    }

    /// Jaccard similarity of two axes' gene sets.
    pub fn jaccard(&self, other: &AxisDefinition) -> f64 {
        use std::collections::HashSet;
        let a: HashSet<&String> = self.gene_ids.iter().collect();
        let b: HashSet<&String> = other.gene_ids.iter().collect();
        let inter = a.intersection(&b).count();
        let uni = a.union(&b).count();
        if uni == 0 {
            0.0
        } else {
            inter as f64 / uni as f64
        }
    }
}

pub struct AxisDefinitionBuilder {
    axis_id: String,
    version: u32,
    gene_ids: Vec<String>,
    weights: Option<Vec<f32>>,
    reduction_method: ReductionMethod,
    default_normalization: NormalizationFlags,
}

impl AxisDefinitionBuilder {
    pub fn with_weights(mut self, weights: Vec<f32>) -> Self {
        self.weights = Some(weights);
        self
    }
    pub fn with_normalization(mut self, flags: NormalizationFlags) -> Self {
        self.default_normalization = flags;
        self
    }
    pub fn build(self) -> Result<AxisDefinition, FieldError> {
        AxisDefinition::new(
            self.axis_id,
            self.version,
            self.gene_ids,
            self.weights,
            self.reduction_method,
            self.default_normalization,
        )
    }
}
