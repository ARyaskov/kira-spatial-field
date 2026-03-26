use std::collections::HashMap;

use crate::{
    axis::AxisDefinition, error::FieldError, field::Field, metadata::FieldMetadata,
    normalization::NormalizationFlags, reduction::ReductionMethod,
};

/// Read-only view over an external CSR expression matrix.
///
/// Implement this trait for external matrix containers that expose CSR arrays
/// and deterministic gene-id to column-index mapping.
pub trait ExpressionCsrView {
    fn indptr(&self) -> &[usize];
    fn indices(&self) -> &[usize];
    fn data(&self) -> &[f32];
    fn gene_index(&self) -> &HashMap<String, usize>;
}

/// Read-only view over an external spatial domain descriptor.
pub trait SpatialDomainView {
    fn id(&self) -> u64;
    fn bin_count(&self) -> usize;
}

/// Opaque CSR matrix interface expected by [`Field::from_gene`].
pub type ExpressionCsr = dyn ExpressionCsrView;
/// Opaque domain interface expected by [`Field::from_gene`].
pub type SpatialDomain = dyn SpatialDomainView;

fn validate_domain_alignment(
    csr: &ExpressionCsr,
    domain: &SpatialDomain,
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

fn row_value_for_gene(
    indices: &[usize],
    data: &[f32],
    start: usize,
    end: usize,
    gene_col: usize,
) -> Result<f32, FieldError> {
    if start > end || end > indices.len() || end > data.len() {
        return Err(FieldError::InvalidMetadata);
    }

    for pos in start..end {
        if indices[pos] == gene_col {
            let observed = data[pos];
            if !observed.is_finite() {
                return Err(FieldError::InvalidValues);
            }
            return Ok(observed);
        }
    }

    Ok(0.0)
}

impl Field {
    /// Builds a deterministic dense single-gene field from CSR input.
    ///
    /// Determinism guarantees:
    /// - Bins are iterated in strictly increasing index order.
    /// - CSR row access uses fixed `indptr[bin]..indptr[bin + 1]` slices.
    /// - Missing gene values are represented as exact `0.0`.
    /// - No implicit transforms, normalization, or floating accumulation occur.
    pub fn from_gene(
        csr: &ExpressionCsr,
        gene_id: &str,
        domain: &SpatialDomain,
    ) -> Result<Self, FieldError> {
        validate_domain_alignment(csr, domain)?;

        let Some(&gene_col) = csr.gene_index().get(gene_id) else {
            return Err(FieldError::InvalidMetadata);
        };

        let mut values = Vec::with_capacity(domain.bin_count());
        let indices = csr.indices();
        let data = csr.data();
        let indptr = csr.indptr();

        for bin in 0..domain.bin_count() {
            let start = indptr[bin];
            let end = indptr[bin + 1];
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

    /// Builds a deterministic dense panel field from CSR input.
    ///
    /// Determinism guarantees:
    /// - Bins are iterated in fixed increasing index order.
    /// - Genes are iterated in the exact input order.
    /// - Reduction is sequential with no parallel aggregation.
    /// - Trimmed-mean sorting uses stable ordering.
    /// - Weighted mode uses sequential linear accumulation in gene order.
    /// - Missing gene values are represented as exact `0.0`.
    /// - No implicit normalization or hidden transforms are applied.
    pub fn from_panel(
        csr: &ExpressionCsr,
        gene_ids: &[String],
        reduction: ReductionMethod,
        weights: Option<&[f32]>,
        domain: &SpatialDomain,
    ) -> Result<Self, FieldError> {
        validate_domain_alignment(csr, domain)?;

        if gene_ids.is_empty() {
            return Err(FieldError::InvalidMetadata);
        }

        if !gene_ids.windows(2).all(|window| window[0] <= window[1]) {
            return Err(FieldError::InvalidMetadata);
        }
        if gene_ids.windows(2).any(|window| window[0] == window[1]) {
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

        let mut gene_columns = Vec::with_capacity(gene_ids.len());
        for gene_id in gene_ids {
            let Some(&gene_col) = csr.gene_index().get(gene_id) else {
                return Err(FieldError::InvalidMetadata);
            };
            gene_columns.push(gene_col);
        }

        let mut values = Vec::with_capacity(domain.bin_count());
        let indices = csr.indices();
        let data = csr.data();
        let indptr = csr.indptr();
        let mut per_gene_values = Vec::with_capacity(gene_ids.len());

        for bin in 0..domain.bin_count() {
            let start = indptr[bin];
            let end = indptr[bin + 1];
            per_gene_values.clear();

            for &gene_col in &gene_columns {
                per_gene_values.push(row_value_for_gene(indices, data, start, end, gene_col)?);
            }

            let reduced = match &reduction {
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
                    let mut sorted = per_gene_values.clone();
                    sorted.sort_by(f32::total_cmp);
                    let n = sorted.len();
                    let trim_each_side = (trim * n as f32).floor() as usize;
                    let begin = trim_each_side;
                    let end_exclusive = n.saturating_sub(trim_each_side);
                    if begin >= end_exclusive {
                        0.0
                    } else {
                        let mut sum = 0.0_f32;
                        for value in &sorted[begin..end_exclusive] {
                            sum += *value;
                        }
                        sum / (end_exclusive - begin) as f32
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

        let metadata = match reduction {
            ReductionMethod::Weighted => FieldMetadata::new(
                "panel::weighted".to_string(),
                gene_ids.to_vec(),
                ReductionMethod::Weighted,
                NormalizationFlags::default(),
                None,
            ),
            _ => FieldMetadata::new(
                "panel::<hashless>".to_string(),
                gene_ids.to_vec(),
                reduction,
                NormalizationFlags::default(),
                None,
            ),
        };

        Self::from_parts(domain.id(), values, metadata).finalize_creation_hash()
    }

    /// Builds a deterministic field from an immutable [`AxisDefinition`].
    ///
    /// Determinism guarantees:
    /// - Axis definition content is immutable and hash-addressed.
    /// - Reduction delegates to fixed-order panel reduction.
    /// - Normalization is applied in strict fixed order.
    /// - No hidden transforms and no parallel execution.
    /// - Identical axis, CSR, and domain inputs yield identical output.
    pub fn from_axis(
        axis: &AxisDefinition,
        csr: &ExpressionCsr,
        domain: &SpatialDomain,
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
                return Err(FieldError::InvalidMetadata);
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
