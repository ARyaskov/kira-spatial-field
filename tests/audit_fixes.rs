use std::collections::HashMap;

use kira_spatial_field::{
    AxisDefinition, Field, FieldError, FieldMetadata, NormalizationFlags, PanelReduction,
    ReductionMethod,
    gene_field::{ExpressionCsrView, SpatialDomainView},
};

// ---- Test fixtures ----

struct MockCsr {
    indptr: Vec<u32>,
    indices: Vec<u32>,
    data: Vec<f32>,
    gene_index: HashMap<String, usize>,
}

impl ExpressionCsrView for MockCsr {
    fn indptr(&self) -> &[u32] {
        &self.indptr
    }
    fn indices(&self) -> &[u32] {
        &self.indices
    }
    fn data(&self) -> &[f32] {
        &self.data
    }
    fn gene_index(&self) -> &HashMap<String, usize> {
        &self.gene_index
    }
}

struct MockDomain {
    id: u64,
    bin_count: usize,
}

impl SpatialDomainView for MockDomain {
    fn id(&self) -> u64 {
        self.id
    }
    fn bin_count(&self) -> usize {
        self.bin_count
    }
}

fn make_csr() -> (MockCsr, MockDomain) {
    // 3 bins × 4 genes
    let mut gene_index = HashMap::new();
    gene_index.insert("A".to_string(), 0);
    gene_index.insert("B".to_string(), 1);
    gene_index.insert("C".to_string(), 2);
    gene_index.insert("D".to_string(), 3);
    let csr = MockCsr {
        indptr: vec![0, 2, 4, 6],
        indices: vec![0, 2, 1, 3, 0, 2], // sorted within each row
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        gene_index,
    };
    let domain = MockDomain {
        id: 42,
        bin_count: 3,
    };
    (csr, domain)
}

// ---- GeneNotFound error variant ----

#[test]
fn from_gene_returns_gene_not_found_on_missing_gene() {
    let (csr, domain) = make_csr();
    let result = Field::from_gene(&csr, "MISSING_GENE", &domain);
    match result {
        Err(FieldError::GeneNotFound { gene_id }) => assert_eq!(gene_id, "MISSING_GENE"),
        other => panic!("expected GeneNotFound, got {:?}", other),
    }
}

#[test]
fn from_panel_returns_gene_not_found_with_specific_id() {
    let (csr, domain) = make_csr();
    let panel = vec!["A".to_string(), "MISSING".to_string()];
    let result = Field::from_panel(&csr, &panel, ReductionMethod::Mean, None, &domain);
    match result {
        Err(FieldError::GeneNotFound { gene_id }) => assert_eq!(gene_id, "MISSING"),
        other => panic!("expected GeneNotFound, got {:?}", other),
    }
}

// ---- from_gene / from_panel correctness ----

#[test]
fn from_gene_returns_correct_values_after_binary_search_change() {
    let (csr, domain) = make_csr();
    // Gene A is at column 0. Bins: 0→col0=1.0, 1→col0=missing→0.0, 2→col0=5.0
    let f = Field::from_gene(&csr, "A", &domain).unwrap();
    assert_eq!(f.values(), &[1.0_f32, 0.0, 5.0]);
}

#[test]
fn from_panel_mean_matches_per_row_computation() {
    let (csr, domain) = make_csr();
    // Panel = [A, C]: bins → mean of cols [0, 2]
    // bin 0: (1.0 + 2.0) / 2 = 1.5
    // bin 1: (0.0 + 0.0) / 2 = 0.0  (neither A nor C in row 1)
    // bin 2: (5.0 + 6.0) / 2 = 5.5
    let panel = vec!["A".to_string(), "C".to_string()];
    let f = Field::from_panel(&csr, &panel, ReductionMethod::Mean, None, &domain).unwrap();
    assert_eq!(f.values(), &[1.5_f32, 0.0, 5.5]);
}

#[test]
fn from_panel_weighted_matches_per_row_computation() {
    let (csr, domain) = make_csr();
    let panel = vec!["A".to_string(), "C".to_string()];
    let weights = vec![1.0_f32, 2.0];
    let f = Field::from_panel(&csr, &panel, ReductionMethod::Weighted, Some(&weights), &domain)
        .unwrap();
    // bin 0: 1.0*1.0 + 2.0*2.0 = 5.0
    // bin 1: 0.0*1.0 + 0.0*2.0 = 0.0
    // bin 2: 5.0*1.0 + 6.0*2.0 = 17.0
    assert_eq!(f.values(), &[5.0_f32, 0.0, 17.0]);
}

// ---- f64 Welford z-score reduces precision loss ----

#[test]
fn zscore_global_uses_high_precision_accumulator() {
    // Large near-mean values where f32 sum loses precision. After f64
    // Welford, mean should be very close to 1.0e7.
    let n = 1024;
    let mut vals = vec![1.0e7_f32; n];
    vals[0] = 1.0e7 + 1.0;
    vals[1] = 1.0e7 - 1.0;
    let domain = MockDomain {
        id: 1,
        bin_count: n,
    };
    let mut gene_index = HashMap::new();
    gene_index.insert("X".to_string(), 0);
    let csr = MockCsr {
        indptr: (0..=n as u32).collect(),
        indices: vec![0_u32; n],
        data: vals.clone(),
        gene_index,
    };
    let f = Field::from_gene(&csr, "X", &domain).unwrap();
    let z = f.zscore_global().unwrap();
    // Almost-uniform input has near-zero variance → output near 0.
    let max_abs = z.values().iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(max_abs < 100.0, "zscore output spread suggests precision loss: {}", max_abs);
}

// ---- Field::from_values public constructor ----

#[test]
fn field_from_values_round_trips_metadata_and_hash() {
    let metadata = FieldMetadata::builder(
        "custom".to_string(),
        vec!["X".to_string()],
        ReductionMethod::SingleGene,
    )
    .build();
    let f = Field::from_values(42, vec![1.0, 2.0, 3.0], metadata).unwrap();
    assert_eq!(f.values(), &[1.0_f32, 2.0, 3.0]);
    assert_eq!(f.domain_id(), 42);
    assert!(f.metadata().creation_hash().is_some());
}

// ---- Field::len / is_empty ----

#[test]
fn field_len_and_is_empty() {
    let (csr, domain) = make_csr();
    let f = Field::from_gene(&csr, "A", &domain).unwrap();
    assert_eq!(f.len(), 3);
    assert!(!f.is_empty());
}

// ---- Field::apply ----

#[test]
fn field_apply_runs_transform_left_to_right() {
    let (csr, domain) = make_csr();
    let f = Field::from_gene(&csr, "A", &domain).unwrap();
    let doubled = f.apply("::double", |v| v * 2.0).unwrap();
    assert_eq!(doubled.values(), &[2.0_f32, 0.0, 10.0]);
    // Provenance: name suffix appended.
    assert!(doubled.metadata().field_name().ends_with("::double"));
}

#[test]
fn field_apply_rejects_non_finite_output() {
    let (csr, domain) = make_csr();
    let f = Field::from_gene(&csr, "A", &domain).unwrap();
    let result = f.apply("::nan", |_v| f32::NAN);
    assert!(matches!(result, Err(FieldError::InvalidValues)));
}

// ---- Field PartialEq via hash ----

#[test]
fn field_equality_via_creation_hash() {
    let (csr, domain) = make_csr();
    let f1 = Field::from_gene(&csr, "A", &domain).unwrap();
    let f2 = Field::from_gene(&csr, "A", &domain).unwrap();
    assert_eq!(f1, f2);

    let f3 = Field::from_gene(&csr, "B", &domain).unwrap();
    assert_ne!(f1, f3);
}

// ---- Field Display ----

#[test]
fn field_display_includes_name_domain_len_hash_prefix() {
    let (csr, domain) = make_csr();
    let f = Field::from_gene(&csr, "A", &domain).unwrap();
    let s = format!("{}", f);
    assert!(s.contains("name=A"));
    assert!(s.contains("domain=42"));
    assert!(s.contains("len=3"));
    assert!(s.contains("hash="));
}

// ---- masked log1p / minmax ----

#[test]
fn log1p_masked_only_transforms_masked_bins() {
    let (csr, domain) = make_csr();
    let f = Field::from_gene(&csr, "A", &domain).unwrap(); // [1.0, 0.0, 5.0]
    let mask = vec![true, false, true];
    let out = f.log1p_masked(&mask).unwrap();
    // bin 0: ln(2) ≈ 0.6931
    // bin 1: unchanged 0.0
    // bin 2: ln(6) ≈ 1.7917
    assert!((out.values()[0] - (1.0_f32 + 1.0).ln()).abs() < 1e-6);
    assert_eq!(out.values()[1], 0.0);
    assert!((out.values()[2] - (1.0_f32 + 5.0).ln()).abs() < 1e-6);
}

#[test]
fn minmax_scale_masked_only_uses_masked_range() {
    let (csr, domain) = make_csr();
    let f = Field::from_gene(&csr, "A", &domain).unwrap();
    let mask = vec![true, false, true];
    let out = f.minmax_scale_masked(&mask).unwrap();
    // Masked range: min=1.0, max=5.0
    // bin 0: (1-1)/(5-1) = 0
    // bin 1: unchanged 0.0
    // bin 2: (5-1)/4 = 1
    assert_eq!(out.values()[0], 0.0);
    assert_eq!(out.values()[1], 0.0);
    assert_eq!(out.values()[2], 1.0);
}

// ---- Field arithmetic ----

#[test]
fn field_addition_combines_per_bin() {
    let (csr, domain) = make_csr();
    let a = Field::from_gene(&csr, "A", &domain).unwrap(); // [1, 0, 5]
    let b = Field::from_gene(&csr, "C", &domain).unwrap(); // [2, 0, 6]
    let sum = a.add(&b).unwrap();
    assert_eq!(sum.values(), &[3.0_f32, 0.0, 11.0]);
}

#[test]
fn field_arithmetic_rejects_domain_mismatch() {
    let (csr, domain) = make_csr();
    let a = Field::from_gene(&csr, "A", &domain).unwrap();
    let other_domain = MockDomain {
        id: 99,
        bin_count: 3,
    };
    let b = Field::from_gene(&csr, "C", &other_domain).unwrap();
    let result = a.add(&b);
    assert!(matches!(result, Err(FieldError::DomainSizeMismatch)));
}

// ---- rank_normalize ----

#[test]
fn rank_normalize_maps_values_to_uniform_zero_one() {
    let metadata = FieldMetadata::builder(
        "ranked".to_string(),
        vec!["X".to_string()],
        ReductionMethod::SingleGene,
    )
    .build();
    let f = Field::from_values(1, vec![3.0, 1.0, 4.0, 1.5, 5.0], metadata).unwrap();
    let r = f.rank_normalize().unwrap();
    // Sorted: 1.0(idx1), 1.5(idx3), 3.0(idx0), 4.0(idx2), 5.0(idx4)
    // Ranks: idx1→0, idx3→1/4, idx0→2/4, idx2→3/4, idx4→4/4
    assert_eq!(r.values()[1], 0.0);
    assert_eq!(r.values()[3], 0.25);
    assert_eq!(r.values()[0], 0.5);
    assert_eq!(r.values()[2], 0.75);
    assert_eq!(r.values()[4], 1.0);
}

// ---- AxisDefinition builder ----

#[test]
fn axis_definition_builder_chains_optional_fields() {
    let axis = AxisDefinition::builder(
        "test_axis",
        1,
        vec!["A".to_string(), "B".to_string()],
        ReductionMethod::Mean,
    )
    .with_normalization(NormalizationFlags {
        log1p: true,
        ..Default::default()
    })
    .build()
    .unwrap();
    assert_eq!(axis.axis_id(), "test_axis");
    assert!(axis.default_normalization().log1p);
}

// ---- AxisDefinition::merge_union and jaccard ----

#[test]
fn axis_merge_union_combines_gene_sets() {
    let a = AxisDefinition::builder(
        "A",
        1,
        vec!["G1".to_string(), "G2".to_string()],
        ReductionMethod::Mean,
    )
    .build()
    .unwrap();
    let b = AxisDefinition::builder(
        "B",
        1,
        vec!["G2".to_string(), "G3".to_string()],
        ReductionMethod::Mean,
    )
    .build()
    .unwrap();
    let merged = a.merge_union(&b, "composite", 1).unwrap();
    let mut got: Vec<String> = merged.gene_ids().to_vec();
    got.sort();
    assert_eq!(got, vec!["G1", "G2", "G3"]);
}

#[test]
fn axis_jaccard_is_intersection_over_union() {
    let a = AxisDefinition::builder(
        "A",
        1,
        vec!["G1".to_string(), "G2".to_string(), "G3".to_string()],
        ReductionMethod::Mean,
    )
    .build()
    .unwrap();
    let b = AxisDefinition::builder(
        "B",
        1,
        vec!["G2".to_string(), "G3".to_string(), "G4".to_string()],
        ReductionMethod::Mean,
    )
    .build()
    .unwrap();
    // intersection: {G2, G3} (2), union: {G1..G4} (4), Jaccard = 0.5
    assert!((a.jaccard(&b) - 0.5).abs() < 1e-9);
}

// ---- PanelReduction <-> ReductionMethod conversion ----

#[test]
fn panel_reduction_into_reduction_method() {
    let pr = PanelReduction::Mean;
    let rm: ReductionMethod = pr.into();
    assert!(matches!(rm, ReductionMethod::Mean));
}

#[test]
fn panel_reduction_from_reduction_method_rejects_single_gene() {
    let result: Result<PanelReduction, _> = ReductionMethod::SingleGene.try_into();
    assert!(matches!(result, Err(FieldError::InvalidReduction)));
}

// ---- FieldErrorExt::with_context ----

#[test]
fn field_error_ext_attaches_context() {
    use kira_spatial_field::FieldErrorExt;
    let r: Result<(), FieldError> = Err(FieldError::InvalidValues);
    let chained = r.with_context("during test");
    let err = chained.unwrap_err();
    let s = format!("{}", err);
    assert!(s.starts_with("during test:"));
}

// ---- from_gene works with generic ExpressionCsrView ----

#[test]
fn from_gene_works_as_generic_function() {
    let (csr, domain) = make_csr();
    // Compiles: no `&dyn` here, MockCsr binds the generic directly.
    let f = Field::from_gene::<MockCsr, MockDomain>(&csr, "B", &domain).unwrap();
    // Gene B at col 1; only bin 1 has it (value 3.0)
    assert_eq!(f.values(), &[0.0_f32, 3.0, 0.0]);
}

// ---- from_panel_cols (pre-resolved column indices) ----

#[test]
fn from_panel_cols_matches_from_panel() {
    let (csr, domain) = make_csr();
    let panel: Vec<String> = vec!["A".into(), "C".into()];

    // Reference: resolve through gene_index.
    let via_panel = Field::from_panel(&csr, &panel, ReductionMethod::Mean, None, &domain).unwrap();

    // Hot path: caller supplies pre-resolved column indices.
    let cols: Vec<u32> = panel.iter().map(|g| csr.gene_index()[g] as u32).collect();
    let via_cols =
        Field::from_panel_cols(&csr, &cols, &panel, ReductionMethod::Mean, None, &domain).unwrap();

    // Both paths must produce bitwise-identical values and hash.
    assert_eq!(via_panel.values(), via_cols.values());
    assert_eq!(via_panel.metadata().creation_hash(), via_cols.metadata().creation_hash());
}

#[test]
fn from_panel_cols_rejects_duplicate_columns() {
    let (csr, domain) = make_csr();
    let result = Field::from_panel_cols(
        &csr,
        &[0_u32, 0_u32],
        &["A".to_string(), "B".to_string()],
        ReductionMethod::Mean,
        None,
        &domain,
    );
    assert!(matches!(result, Err(FieldError::InvalidMetadata)));
}

#[test]
fn from_panel_cols_rejects_unsorted_names() {
    let (csr, domain) = make_csr();
    let result = Field::from_panel_cols(
        &csr,
        &[2_u32, 0_u32],
        &["C".to_string(), "A".to_string()],
        ReductionMethod::Mean,
        None,
        &domain,
    );
    assert!(matches!(result, Err(FieldError::InvalidMetadata)));
}

// ---- axis_expressing_mask ----

#[test]
fn axis_expressing_mask_flags_bins_with_axis_genes() {
    let (csr, domain) = make_csr();
    // CSR: bin 0 → {A=1, C=2}; bin 1 → {B=3, D=4}; bin 2 → {A=5, C=6}.
    let axis = AxisDefinition::builder(
        "ac",
        1,
        vec!["A".to_string(), "C".to_string()],
        ReductionMethod::Mean,
    )
    .build()
    .unwrap();
    let mask = kira_spatial_field::axis_expressing_mask(&axis, &csr, &domain).unwrap();
    assert_eq!(mask, vec![true, false, true]);
}

#[test]
fn axis_expressing_mask_returns_gene_not_found_on_missing_gene() {
    let (csr, domain) = make_csr();
    let axis = AxisDefinition::builder(
        "missing",
        1,
        vec!["NOT_IN_CSR".to_string()],
        ReductionMethod::Mean,
    )
    .build()
    .unwrap();
    let result = kira_spatial_field::axis_expressing_mask(&axis, &csr, &domain);
    match result {
        Err(FieldError::GeneNotFound { gene_id }) => assert_eq!(gene_id, "NOT_IN_CSR"),
        other => panic!("expected GeneNotFound, got {:?}", other.map(|_| ())),
    }
}
