use std::collections::HashMap;

use crate::{
    axis::AxisDefinition, error::FieldError, field::Field, metadata::FieldMetadata,
    normalization::NormalizationFlags, reduction::ReductionMethod,
};

/// Read-only view over an external CSR expression matrix.
pub trait ExpressionCsrView {
    fn indptr(&self) -> &[u32];
    fn indices(&self) -> &[u32];
    fn data(&self) -> &[f32];
    fn gene_index(&self) -> &HashMap<String, usize>;
}

/// Read-only view over an external spatial domain descriptor.
pub trait SpatialDomainView {
    fn id(&self) -> u64;
    fn bin_count(&self) -> usize;
}

/// `dyn`-erased alias for [`ExpressionCsrView`]. Prefer the generic form
/// on hot paths to avoid vtable indirection.
pub type ExpressionCsr = dyn ExpressionCsrView;
pub type SpatialDomain = dyn SpatialDomainView;

fn validate_domain_alignment<C: ExpressionCsrView + ?Sized, D: SpatialDomainView + ?Sized>(
    csr: &C,
    domain: &D,
) -> Result<(), FieldError> {
    let row_count = csr
        .indptr()
        .len()
        .checked_sub(1)
        .ok_or(FieldError::DomainSizeMismatch)?;
    if domain.bin_count() != row_count {
        return Err(FieldError::DomainSizeMismatch);
    }
    Ok(())
}

/// Shared panel-reduction loop. Caller must validate `gene_columns`
/// (non-empty, no duplicates) and reduction parameters.
fn panel_reduce_core<C: ExpressionCsrView + ?Sized, D: SpatialDomainView + ?Sized>(
    csr: &C,
    gene_columns: &[u32],
    reduction: &ReductionMethod,
    weights: Option<&[f32]>,
    trim_fraction: Option<f32>,
    domain: &D,
) -> Result<Vec<f32>, FieldError> {
    let mut col_to_pos: Vec<(u32, usize)> = gene_columns
        .iter()
        .enumerate()
        .map(|(panel_pos, &col)| (col, panel_pos))
        .collect();
    col_to_pos.sort_unstable_by_key(|&(col, _)| col);
    let sorted_cols: Vec<u32> = col_to_pos.iter().map(|&(c, _)| c).collect();

    let mut values = Vec::with_capacity(domain.bin_count());
    let indices = csr.indices();
    let data = csr.data();
    let indptr = csr.indptr();
    let mut per_gene_values = vec![0.0_f32; gene_columns.len()];

    for bin in 0..domain.bin_count() {
        let start = indptr[bin] as usize;
        let end = indptr[bin + 1] as usize;
        if start > end || end > indices.len() || end > data.len() {
            return Err(FieldError::InvalidMetadata);
        }

        for v in per_gene_values.iter_mut() {
            *v = 0.0;
        }
        for p in start..end {
            let row_col = indices[p];
            if let Ok(idx_in_sorted) = sorted_cols.binary_search(&row_col) {
                let panel_pos = col_to_pos[idx_in_sorted].1;
                let observed = data[p];
                if !observed.is_finite() {
                    return Err(FieldError::InvalidValues);
                }
                per_gene_values[panel_pos] = observed;
            }
        }

        let reduced = match reduction {
            ReductionMethod::Mean => {
                let mut sum = 0.0_f32;
                for value in &per_gene_values {
                    sum += *value;
                }
                sum / per_gene_values.len() as f32
            }
            ReductionMethod::TrimmedMean { .. } => {
                let Some(trim) = trim_fraction else {
                    return Err(FieldError::InvalidReduction);
                };
                let mut scratch = per_gene_values.clone();
                let n = scratch.len();
                let trim_each_side = (trim * n as f32).floor() as usize;
                let begin = trim_each_side;
                let end_exclusive = n.saturating_sub(trim_each_side);
                if begin >= end_exclusive {
                    0.0
                } else {
                    if begin > 0 {
                        scratch.select_nth_unstable_by(begin - 1, f32::total_cmp);
                    }
                    if end_exclusive < n {
                        scratch[begin..]
                            .select_nth_unstable_by(end_exclusive - begin, f32::total_cmp);
                    }
                    let mut sum = 0.0_f64;
                    for value in &scratch[begin..end_exclusive] {
                        sum += *value as f64;
                    }
                    (sum / (end_exclusive - begin) as f64) as f32
                }
            }
            ReductionMethod::Weighted => {
                let Some(weight_values) = weights else {
                    return Err(FieldError::InvalidReduction);
                };
                let mut sum = 0.0_f32;
                for i in 0..per_gene_values.len() {
                    sum += per_gene_values[i] * weight_values[i];
                }
                sum
            }
            ReductionMethod::SingleGene => return Err(FieldError::InvalidReduction),
        };

        if !reduced.is_finite() {
            return Err(FieldError::InvalidValues);
        }
        values.push(reduced);
    }
    Ok(values)
}

fn row_value_for_gene(
    indices: &[u32],
    data: &[f32],
    start: usize,
    end: usize,
    gene_col: u32,
) -> Result<f32, FieldError> {
    if start > end || end > indices.len() || end > data.len() {
        return Err(FieldError::InvalidMetadata);
    }

    // CSR per-row indices are sorted ascending — binary search is safe.
    let row_indices = &indices[start..end];
    match row_indices.binary_search(&gene_col) {
        Ok(pos) => {
            let observed = data[start + pos];
            if !observed.is_finite() {
                Err(FieldError::InvalidValues)
            } else {
                Ok(observed)
            }
        }
        Err(_) => Ok(0.0),
    }
}

impl Field {
    /// Builds a deterministic dense single-gene field from CSR input.
    /// Missing gene values are represented as exact `0.0`.
    pub fn from_gene<C: ExpressionCsrView + ?Sized, D: SpatialDomainView + ?Sized>(
        csr: &C,
        gene_id: &str,
        domain: &D,
    ) -> Result<Self, FieldError> {
        validate_domain_alignment(csr, domain)?;

        let Some(&gene_col_usize) = csr.gene_index().get(gene_id) else {
            return Err(FieldError::GeneNotFound {
                gene_id: gene_id.to_string(),
            });
        };
        let gene_col: u32 = gene_col_usize
            .try_into()
            .map_err(|_| FieldError::InvalidMetadata)?;

        let mut values = Vec::with_capacity(domain.bin_count());
        let indices = csr.indices();
        let data = csr.data();
        let indptr = csr.indptr();

        for bin in 0..domain.bin_count() {
            let start = indptr[bin] as usize;
            let end = indptr[bin + 1] as usize;
            values.push(row_value_for_gene(indices, data, start, end, gene_col)?);
        }

        if values.iter().any(|value| !value.is_finite()) {
            return Err(FieldError::InvalidValues);
        }

        let metadata = FieldMetadata::new(
            gene_id.to_string(),
            vec![gene_id.to_string()],
            ReductionMethod::SingleGene,
            NormalizationFlags::default(),
            None,
        );

        Self::from_parts(domain.id(), values, metadata).finalize_creation_hash()
    }

    /// Builds a deterministic dense panel field from CSR input. `gene_ids`
    /// must be sorted ascending and contain no duplicates.
    pub fn from_panel<
        C: ExpressionCsrView + ?Sized,
        D: SpatialDomainView + ?Sized,
        S: AsRef<str>,
    >(
        csr: &C,
        gene_ids: &[S],
        reduction: ReductionMethod,
        weights: Option<&[f32]>,
        domain: &D,
    ) -> Result<Self, FieldError> {
        validate_domain_alignment(csr, domain)?;

        if gene_ids.is_empty() {
            return Err(FieldError::InvalidMetadata);
        }

        if !gene_ids.windows(2).all(|w| w[0].as_ref() <= w[1].as_ref()) {
            return Err(FieldError::InvalidMetadata);
        }
        if gene_ids.windows(2).any(|w| w[0].as_ref() == w[1].as_ref()) {
            return Err(FieldError::InvalidMetadata);
        }

        let trim_fraction = match &reduction {
            ReductionMethod::Mean => {
                if weights.is_some() {
                    return Err(FieldError::InvalidReduction);
                }
                None
            }
            ReductionMethod::TrimmedMean { trim_fraction } => {
                if weights.is_some() {
                    return Err(FieldError::InvalidReduction);
                }
                if !(*trim_fraction >= 0.0 && *trim_fraction < 0.5) {
                    return Err(FieldError::InvalidReduction);
                }
                Some(*trim_fraction)
            }
            ReductionMethod::Weighted => {
                let Some(weight_values) = weights else {
                    return Err(FieldError::InvalidReduction);
                };
                if weight_values.len() != gene_ids.len() {
                    return Err(FieldError::InvalidReduction);
                }
                if weight_values.iter().any(|weight| !weight.is_finite()) {
                    return Err(FieldError::InvalidReduction);
                }
                None
            }
            ReductionMethod::SingleGene => return Err(FieldError::InvalidReduction),
        };

        let mut gene_columns: Vec<u32> = Vec::with_capacity(gene_ids.len());
        for gene_id in gene_ids {
            let id_str = gene_id.as_ref();
            let Some(&gene_col_usize) = csr.gene_index().get(id_str) else {
                return Err(FieldError::GeneNotFound {
                    gene_id: id_str.to_string(),
                });
            };
            let gene_col: u32 = gene_col_usize
                .try_into()
                .map_err(|_| FieldError::InvalidMetadata)?;
            gene_columns.push(gene_col);
        }

        let owned_genes: Vec<String> = gene_ids.iter().map(|s| s.as_ref().to_string()).collect();
        let values =
            panel_reduce_core(csr, &gene_columns, &reduction, weights, trim_fraction, domain)?;

        let metadata = match reduction {
            ReductionMethod::Weighted => FieldMetadata::new(
                "panel::weighted".to_string(),
                owned_genes,
                ReductionMethod::Weighted,
                NormalizationFlags::default(),
                None,
            ),
            _ => FieldMetadata::new(
                "panel::<hashless>".to_string(),
                owned_genes,
                reduction,
                NormalizationFlags::default(),
                None,
            ),
        };

        Self::from_parts(domain.id(), values, metadata).finalize_creation_hash()
    }

    /// Fast-path companion to [`Field::from_panel`] taking pre-resolved
    /// CSR column indices. `gene_id_names` (sorted ascending, no duplicates)
    /// is used for metadata provenance.
    pub fn from_panel_cols<
        C: ExpressionCsrView + ?Sized,
        D: SpatialDomainView + ?Sized,
        S: AsRef<str>,
    >(
        csr: &C,
        gene_columns: &[u32],
        gene_id_names: &[S],
        reduction: ReductionMethod,
        weights: Option<&[f32]>,
        domain: &D,
    ) -> Result<Self, FieldError> {
        validate_domain_alignment(csr, domain)?;

        if gene_columns.is_empty() {
            return Err(FieldError::InvalidMetadata);
        }
        if gene_columns.len() != gene_id_names.len() {
            return Err(FieldError::InvalidMetadata);
        }
        if !gene_id_names
            .windows(2)
            .all(|w| w[0].as_ref() <= w[1].as_ref())
        {
            return Err(FieldError::InvalidMetadata);
        }
        if gene_id_names
            .windows(2)
            .any(|w| w[0].as_ref() == w[1].as_ref())
        {
            return Err(FieldError::InvalidMetadata);
        }
        let mut sorted_check: Vec<u32> = gene_columns.to_vec();
        sorted_check.sort_unstable();
        if sorted_check.windows(2).any(|w| w[0] == w[1]) {
            return Err(FieldError::InvalidMetadata);
        }

        let trim_fraction = match &reduction {
            ReductionMethod::Mean => {
                if weights.is_some() {
                    return Err(FieldError::InvalidReduction);
                }
                None
            }
            ReductionMethod::TrimmedMean { trim_fraction } => {
                if weights.is_some() {
                    return Err(FieldError::InvalidReduction);
                }
                if !(*trim_fraction >= 0.0 && *trim_fraction < 0.5) {
                    return Err(FieldError::InvalidReduction);
                }
                Some(*trim_fraction)
            }
            ReductionMethod::Weighted => {
                let Some(weight_values) = weights else {
                    return Err(FieldError::InvalidReduction);
                };
                if weight_values.len() != gene_columns.len() {
                    return Err(FieldError::InvalidReduction);
                }
                if weight_values.iter().any(|weight| !weight.is_finite()) {
                    return Err(FieldError::InvalidReduction);
                }
                None
            }
            ReductionMethod::SingleGene => return Err(FieldError::InvalidReduction),
        };

        let owned_genes: Vec<String> = gene_id_names
            .iter()
            .map(|s| s.as_ref().to_string())
            .collect();
        let values =
            panel_reduce_core(csr, gene_columns, &reduction, weights, trim_fraction, domain)?;

        let metadata = match reduction {
            ReductionMethod::Weighted => FieldMetadata::new(
                "panel::weighted".to_string(),
                owned_genes,
                ReductionMethod::Weighted,
                NormalizationFlags::default(),
                None,
            ),
            _ => FieldMetadata::new(
                "panel::<hashless>".to_string(),
                owned_genes,
                reduction,
                NormalizationFlags::default(),
                None,
            ),
        };

        Self::from_parts(domain.id(), values, metadata).finalize_creation_hash()
    }

    /// Builds a deterministic field from an immutable [`AxisDefinition`].
    pub fn from_axis<C: ExpressionCsrView + ?Sized, D: SpatialDomainView + ?Sized>(
        axis: &AxisDefinition,
        csr: &C,
        domain: &D,
    ) -> Result<Self, FieldError> {
        validate_domain_alignment(csr, domain)?;

        if axis.gene_ids().is_empty() {
            return Err(FieldError::InvalidMetadata);
        }

        if axis.definition_hash().iter().all(|byte| *byte == 0) {
            return Err(FieldError::InvalidMetadata);
        }

        for gene_id in axis.gene_ids() {
            if !csr.gene_index().contains_key(gene_id) {
                return Err(FieldError::GeneNotFound {
                    gene_id: gene_id.clone(),
                });
            }
        }

        if axis.default_normalization().zscore_masked {
            return Err(FieldError::InvalidReduction);
        }

        let mut field = match axis.reduction_method() {
            ReductionMethod::Mean | ReductionMethod::TrimmedMean { .. } => Self::from_panel(
                csr,
                axis.gene_ids(),
                axis.reduction_method().clone(),
                None,
                domain,
            )?,
            ReductionMethod::Weighted => Self::from_panel(
                csr,
                axis.gene_ids(),
                ReductionMethod::Weighted,
                axis.weights(),
                domain,
            )?,
            ReductionMethod::SingleGene => return Err(FieldError::InvalidReduction),
        };

        let normalization = *axis.default_normalization();
        if normalization.log1p {
            field = field.log1p()?;
        }
        if normalization.zscore_global {
            field = field.zscore_global()?;
        }
        if normalization.minmax_scale {
            field = field.minmax_scale()?;
        }

        let metadata = FieldMetadata::new(
            axis.axis_id().to_string(),
            axis.gene_ids().to_vec(),
            axis.reduction_method().clone(),
            normalization,
            Some(*axis.definition_hash()),
        );

        field.with_metadata(metadata).finalize_creation_hash()
    }
}

/// Boolean mask where `mask[bin] = true` iff at least one of
/// `axis.gene_ids()` has a non-zero CSR entry in that bin.
pub fn axis_expressing_mask<C: ExpressionCsrView + ?Sized, D: SpatialDomainView + ?Sized>(
    axis: &AxisDefinition,
    csr: &C,
    domain: &D,
) -> Result<Vec<bool>, FieldError> {
    validate_domain_alignment(csr, domain)?;

    if axis.gene_ids().is_empty() {
        return Err(FieldError::InvalidMetadata);
    }

    let mut gene_cols: Vec<u32> = Vec::with_capacity(axis.gene_ids().len());
    for gene_id in axis.gene_ids() {
        let Some(&col_usize) = csr.gene_index().get(gene_id) else {
            return Err(FieldError::GeneNotFound {
                gene_id: gene_id.clone(),
            });
        };
        let col: u32 = col_usize
            .try_into()
            .map_err(|_| FieldError::InvalidMetadata)?;
        gene_cols.push(col);
    }
    gene_cols.sort_unstable();
    gene_cols.dedup();

    let indptr = csr.indptr();
    let indices = csr.indices();
    let data = csr.data();

    let mut mask = Vec::with_capacity(domain.bin_count());
    for bin in 0..domain.bin_count() {
        let start = indptr[bin] as usize;
        let end = indptr[bin + 1] as usize;
        if start > end || end > indices.len() || end > data.len() {
            return Err(FieldError::InvalidMetadata);
        }
        let mut hit = false;
        for p in start..end {
            let col = indices[p];
            if gene_cols.binary_search(&col).is_ok() {
                let v = data[p];
                if !v.is_finite() {
                    return Err(FieldError::InvalidValues);
                }
                if v != 0.0 {
                    hit = true;
                    break;
                }
            }
        }
        mask.push(hit);
    }
    Ok(mask)
}
