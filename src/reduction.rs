use crate::error::FieldError;
use serde::{Deserialize, Serialize};

/// Deterministic scalar-reduction strategy over selected genes.
/// Discriminants are part of the hash contract — do not reorder.
#[repr(u8)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ReductionMethod {
    SingleGene = 0,
    Mean = 1,
    TrimmedMean { trim_fraction: f32 } = 2,
    Weighted = 3,
}

impl ReductionMethod {
    pub const fn discriminant(&self) -> u8 {
        match self {
            Self::SingleGene => 0,
            Self::Mean => 1,
            Self::TrimmedMean { .. } => 2,
            Self::Weighted => 3,
        }
    }

    pub fn hash_payload(&self) -> Option<[u8; 4]> {
        match self {
            Self::TrimmedMean { trim_fraction } => Some(trim_fraction.to_le_bytes()),
            _ => None,
        }
    }
}

/// Reduction strategy restricted to panel (multi-gene) modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PanelReduction {
    Mean,
    TrimmedMean { trim_fraction: f32 },
    Weighted,
}

impl From<PanelReduction> for ReductionMethod {
    fn from(value: PanelReduction) -> Self {
        match value {
            PanelReduction::Mean => ReductionMethod::Mean,
            PanelReduction::TrimmedMean { trim_fraction } => {
                ReductionMethod::TrimmedMean { trim_fraction }
            }
            PanelReduction::Weighted => ReductionMethod::Weighted,
        }
    }
}

impl TryFrom<ReductionMethod> for PanelReduction {
    type Error = FieldError;
    fn try_from(value: ReductionMethod) -> Result<Self, Self::Error> {
        match value {
            ReductionMethod::Mean => Ok(PanelReduction::Mean),
            ReductionMethod::TrimmedMean { trim_fraction } => {
                Ok(PanelReduction::TrimmedMean { trim_fraction })
            }
            ReductionMethod::Weighted => Ok(PanelReduction::Weighted),
            ReductionMethod::SingleGene => Err(FieldError::InvalidReduction),
        }
    }
}

/// Seurat-style module score over a gene panel.
///
/// This is a deterministic implementation of the standard "signature / module
/// score" used in single-cell and spatial transcriptomics (cf. Seurat's
/// `AddModuleScore`, scanpy's `score_genes`): each gene is z-scored **across
/// bins** (so genes contribute equally regardless of absolute expression), and
/// the per-bin score is the mean of the panel's z-scores. This is the correct
/// way to aggregate a biologically heterogeneous marker panel — unlike summing
/// raw counts, which is dominated by the highest-expression gene and adds
/// anti-correlated cell-type markers together.
///
/// - `per_gene[g]` is the dense, per-bin expression vector for gene `g`
///   (already normalized, e.g. CPM + log1p). All vectors must have length `n`.
/// - `valid[i]` marks bins included in the per-gene mean/std and in the output;
///   invalid bins are set to `0.0`.
///
/// Statistics use f64 Welford accumulation in fixed gene/bin order — no parallel
/// floating-point reduction — so the result is bitwise-reproducible. A gene with
/// zero variance over valid bins contributes `0.0` (it carries no information).
pub fn module_score(per_gene: &[&[f32]], valid: &[bool]) -> Result<Vec<f32>, FieldError> {
    if per_gene.is_empty() {
        return Err(FieldError::InvalidValues);
    }
    let n = valid.len();
    for g in per_gene {
        if g.len() != n {
            return Err(FieldError::DomainSizeMismatch);
        }
        if g.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::InvalidValues);
        }
    }

    let n_valid = valid.iter().filter(|&&b| b).count();
    let mut acc = vec![0.0_f64; n];
    if n_valid == 0 {
        return Ok(vec![0.0_f32; n]);
    }

    for gene in per_gene {
        // f64 Welford over valid bins (deterministic, no cancellation).
        let mut count = 0_u64;
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        for (i, &v) in gene.iter().enumerate() {
            if !valid[i] {
                continue;
            }
            let x = v as f64;
            count += 1;
            let delta = x - mean;
            mean += delta / count as f64;
            m2 += delta * (x - mean);
        }
        let var = if count > 0 { m2 / count as f64 } else { 0.0 };
        let sigma = var.sqrt();
        if !(sigma.is_finite()) || sigma == 0.0 {
            continue; // zero-variance gene contributes nothing
        }
        for (i, &v) in gene.iter().enumerate() {
            if valid[i] {
                acc[i] += (v as f64 - mean) / sigma;
            }
        }
    }

    let inv_g = 1.0 / per_gene.len() as f64;
    let mut out = vec![0.0_f32; n];
    for (i, slot) in out.iter_mut().enumerate() {
        if valid[i] {
            let s = acc[i] * inv_g;
            *slot = if s.is_finite() { s as f32 } else { 0.0 };
        }
    }
    Ok(out)
}

#[cfg(test)]
mod module_score_tests {
    use super::module_score;

    #[test]
    fn equal_weight_regardless_of_scale() {
        // Gene A in [0,1000] scale, gene B in [0,1] scale, same spatial pattern.
        let a = [0.0_f32, 1000.0, 2000.0, 3000.0];
        let b = [0.0_f32, 1.0, 2.0, 3.0];
        let valid = [true, true, true, true];
        let s = module_score(&[&a, &b], &valid).unwrap();
        // Both genes z-score identically, so the score equals each gene's z-score.
        // Monotonic increasing, mean ~0.
        assert!(s[0] < s[1] && s[1] < s[2] && s[2] < s[3]);
        let mean: f32 = s.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "module score should be ~mean-zero: {mean}");
        // A high-scale raw sum would be dominated by gene A; here both contribute equally.
        assert!((s[3] - 1.3416).abs() < 1e-2, "z of last elem ~1.34: {}", s[3]);
    }

    #[test]
    fn zero_variance_gene_is_ignored() {
        let flat = [5.0_f32, 5.0, 5.0, 5.0];
        let ramp = [0.0_f32, 1.0, 2.0, 3.0];
        let valid = [true, true, true, true];
        let s = module_score(&[&flat, &ramp], &valid).unwrap();
        // Score is just the ramp's z-score (flat gene ignored), so monotone.
        assert!(s[0] < s[3]);
    }

    #[test]
    fn invalid_bins_are_zero() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let valid = [true, false, true, true];
        let s = module_score(&[&a], &valid).unwrap();
        assert_eq!(s[1], 0.0);
    }
}
