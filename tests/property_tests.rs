use std::collections::HashMap;

use proptest::prelude::*;

use kira_spatial_field::{
    AxisDefinition, Field, FieldMetadata, NormalizationFlags, ReductionMethod,
    gene_field::{ExpressionCsrView, SpatialDomainView},
};

// ---- Mock CSR / domain ----

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

/// Build a small dense-ish CSR with `bins` rows and `genes` columns.
/// Every row writes one entry at column `(b + offset) % genes` so each
/// bin contributes exactly one non-zero value — keeps strategies fast
/// to shrink while still exercising the per-row binary-search path.
fn build_csr(bins: usize, genes: usize, values: &[f32]) -> (MockCsr, Vec<String>, MockDomain) {
    assert!(values.len() == bins);
    assert!(genes >= 1);

    let gene_ids: Vec<String> = (0..genes).map(|i| format!("G{:03}", i)).collect();
    let gene_index: HashMap<String, usize> = gene_ids
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();

    let mut indptr: Vec<u32> = Vec::with_capacity(bins + 1);
    let mut indices: Vec<u32> = Vec::with_capacity(bins);
    let mut data: Vec<f32> = Vec::with_capacity(bins);
    indptr.push(0);
    for (b, &v) in values.iter().enumerate() {
        let col = (b % genes) as u32;
        indices.push(col);
        data.push(v);
        indptr.push((b + 1) as u32);
    }

    let domain = MockDomain {
        id: 7,
        bin_count: bins,
    };
    (
        MockCsr {
            indptr,
            indices,
            data,
            gene_index,
        },
        gene_ids,
        domain,
    )
}

/// Strategy: a small slice of finite f32 values in a tame range — the
/// rank/order properties care about ordering rather than magnitude, so
/// `-1e3..=1e3` keeps the search space well-behaved.
fn finite_values(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1_000.0_f32..=1_000.0_f32, min_len..=max_len)
}

/// Strategy: non-negative finite f32 values (for log1p, which requires
/// `value > -1`).
fn nonneg_values(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(0.0_f32..=1_000.0_f32, min_len..=max_len)
}

/// Strategy: gene-id-style strings using a tiny alphabet so collisions
/// (and therefore overlaps under merge_union) actually happen during
/// shrinking, and dedup-by-sort yields stable canonical orderings.
fn gene_id() -> impl Strategy<Value = String> {
    "[A-D]{1,2}"
}

fn gene_id_set(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<String>> {
    prop::collection::hash_set(gene_id(), min_len..=max_len)
        .prop_map(|set| set.into_iter().collect())
}

fn build_field(values: Vec<f32>) -> Field {
    let metadata = FieldMetadata::builder(
        "pt".to_string(),
        vec!["G".to_string()],
        ReductionMethod::SingleGene,
    )
    .with_normalization_flags(NormalizationFlags::default())
    .build();
    Field::from_values(7, values, metadata).expect("from_values should succeed for finite input")
}

// ---- Properties ----

proptest! {
    /// `creation_hash` is purely a function of inputs: two identical
    /// constructions yield the exact same digest. This is the
    /// determinism contract advertised in the README.
    #[test]
    fn from_gene_hash_is_deterministic(
        values in finite_values(1, 16),
        gene_count in 1usize..6,
    ) {
        let bins = values.len();
        let (csr, gene_ids, domain) = build_csr(bins, gene_count, &values);
        let gene = &gene_ids[0];

        let f1 = Field::from_gene(&csr, gene, &domain).unwrap();
        let f2 = Field::from_gene(&csr, gene, &domain).unwrap();
        prop_assert_eq!(f1.metadata().creation_hash(), f2.metadata().creation_hash());
        prop_assert_eq!(f1.values(), f2.values());
    }

    /// `rank_normalize` is order-preserving: if `a[i] < a[j]`, then
    /// `rank[i] < rank[j]`. Ties are broken by index — bin `i` with
    /// `i < j` and `a[i] == a[j]` still gets the smaller rank.
    #[test]
    fn rank_normalize_is_monotonic(values in finite_values(2, 32)) {
        let field = build_field(values.clone());
        let ranked = field.rank_normalize().unwrap();
        let r = ranked.values();
        prop_assert_eq!(r.len(), values.len());

        for i in 0..values.len() {
            for j in 0..values.len() {
                if i == j {
                    continue;
                }
                let primary = values[i].total_cmp(&values[j]);
                if primary == std::cmp::Ordering::Less {
                    prop_assert!(r[i] < r[j], "i={} j={} a={:?} b={:?} ri={} rj={}",
                        i, j, values[i], values[j], r[i], r[j]);
                } else if primary == std::cmp::Ordering::Equal && i < j {
                    prop_assert!(r[i] < r[j], "tie i<j should rank lower; i={} j={} r=({}, {})",
                        i, j, r[i], r[j]);
                }
            }
        }
    }

    /// `rank_normalize` output is bounded to `[0, 1]` and the extremes
    /// are exact: smallest bin gets `0.0`, largest gets `1.0` (when the
    /// field has more than one element).
    #[test]
    fn rank_normalize_range_is_unit(values in finite_values(2, 32)) {
        let field = build_field(values);
        let ranked = field.rank_normalize().unwrap();
        let r = ranked.values();
        let mut min = r[0];
        let mut max = r[0];
        for &v in r.iter() {
            prop_assert!(v.is_finite());
            prop_assert!((0.0..=1.0).contains(&v), "rank {} out of [0,1]", v);
            if v < min { min = v; }
            if v > max { max = v; }
        }
        prop_assert_eq!(min, 0.0);
        prop_assert_eq!(max, 1.0);
    }

    /// `log1p_masked` leaves unmasked bins **bitwise-identical** to the
    /// input. Masked-bin invariance is the contract that lets callers
    /// compose masked transforms without aliasing the unmasked region.
    #[test]
    fn log1p_masked_preserves_unmasked_bins(
        values in nonneg_values(2, 24),
        seed in any::<u64>(),
    ) {
        // Deterministically derive a mask from `seed` so the property
        // shrinks against both `values` and the mask shape.
        let mut mask = Vec::with_capacity(values.len());
        let mut s = seed;
        let mut any_true = false;
        for _ in 0..values.len() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b = (s >> 33) & 1 == 1;
            mask.push(b);
            any_true |= b;
        }
        if !any_true {
            mask[0] = true;
        }

        let field = build_field(values.clone());
        let out = field.log1p_masked(&mask).unwrap();
        let o = out.values();
        for i in 0..values.len() {
            if !mask[i] {
                prop_assert_eq!(o[i].to_bits(), values[i].to_bits(),
                    "unmasked bin {} drifted: in={} out={}", i, values[i], o[i]);
            } else {
                let expected = (1.0_f32 + values[i]).ln();
                prop_assert_eq!(o[i].to_bits(), expected.to_bits(),
                    "masked bin {} mismatch: expected={} got={}", i, expected, o[i]);
            }
        }
    }

    /// Same masked-bin invariance for `minmax_scale_masked`. The
    /// scaling uses only masked entries, so unmasked outputs must
    /// equal their inputs bit-for-bit.
    #[test]
    fn minmax_scale_masked_preserves_unmasked_bins(
        values in finite_values(2, 24),
        seed in any::<u64>(),
    ) {
        let mut mask = Vec::with_capacity(values.len());
        let mut s = seed;
        let mut any_true = false;
        for _ in 0..values.len() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b = (s >> 33) & 1 == 1;
            mask.push(b);
            any_true |= b;
        }
        if !any_true {
            mask[0] = true;
        }

        let field = build_field(values.clone());
        let out = field.minmax_scale_masked(&mask).unwrap();
        let o = out.values();
        for i in 0..values.len() {
            if !mask[i] {
                prop_assert_eq!(o[i].to_bits(), values[i].to_bits(),
                    "unmasked bin {} drifted", i);
            } else {
                prop_assert!(o[i].is_finite());
                prop_assert!((0.0..=1.0).contains(&o[i]),
                    "masked bin {} out of [0,1]: {}", i, o[i]);
            }
        }
    }

    /// `merge_union` is commutative on the gene-id set (the composite
    /// id and metadata may differ, but the resulting alphabetical
    /// gene-id list must not depend on input order).
    #[test]
    fn axis_merge_union_is_commutative(
        ids_a in gene_id_set(1, 6),
        ids_b in gene_id_set(1, 6),
    ) {
        let axis_a = AxisDefinition::new(
            "A".into(),
            1,
            ids_a,
            None,
            ReductionMethod::Mean,
            NormalizationFlags::default(),
        ).unwrap();
        let axis_b = AxisDefinition::new(
            "B".into(),
            1,
            ids_b,
            None,
            ReductionMethod::Mean,
            NormalizationFlags::default(),
        ).unwrap();

        let ab = axis_a.merge_union(&axis_b, "AB", 1).unwrap();
        let ba = axis_b.merge_union(&axis_a, "BA", 1).unwrap();
        prop_assert_eq!(ab.gene_ids(), ba.gene_ids());
    }

    /// Jaccard similarity is symmetric: a.jaccard(b) == b.jaccard(a),
    /// and an axis is self-similar at the unit value.
    #[test]
    fn axis_jaccard_is_symmetric_and_reflexive(
        ids_a in gene_id_set(1, 6),
        ids_b in gene_id_set(1, 6),
    ) {
        let axis_a = AxisDefinition::new(
            "A".into(),
            1,
            ids_a,
            None,
            ReductionMethod::Mean,
            NormalizationFlags::default(),
        ).unwrap();
        let axis_b = AxisDefinition::new(
            "B".into(),
            1,
            ids_b,
            None,
            ReductionMethod::Mean,
            NormalizationFlags::default(),
        ).unwrap();

        // Symmetry — both directions must agree to f64 bit precision
        // because `jaccard` is integer-ratio division of the same
        // intersection/union counts.
        let jab = axis_a.jaccard(&axis_b);
        let jba = axis_b.jaccard(&axis_a);
        prop_assert_eq!(jab.to_bits(), jba.to_bits());
        prop_assert!((0.0..=1.0).contains(&jab));

        // Reflexivity — a non-empty axis is identical to itself.
        let jaa = axis_a.jaccard(&axis_a);
        prop_assert_eq!(jaa, 1.0);
    }

    /// `add` then `sub` on the same operand returns to the original
    /// values within a generous f32 tolerance. f32 arithmetic is not
    /// bitwise-reversible, but it round-trips inside `1e-3 * max(|a|, |b|)`
    /// for the value ranges used here.
    #[test]
    fn add_then_sub_round_trips(values in finite_values(1, 16)) {
        let a = build_field(values.clone());
        let b = build_field(values.clone());
        let summed = a.add(&b).unwrap();
        let recovered = summed.sub(&b).unwrap();
        for (orig, got) in values.iter().zip(recovered.values().iter()) {
            let tol = (orig.abs().max(1.0)) * 1e-3;
            prop_assert!((orig - got).abs() <= tol,
                "add/sub diverged: orig={} got={} tol={}", orig, got, tol);
        }
    }

    /// `Field::from_values` accepts any finite f32 vector and the
    /// resulting field reports identical length and bin-by-bin values.
    #[test]
    fn from_values_round_trips(values in finite_values(1, 32)) {
        let field = build_field(values.clone());
        prop_assert_eq!(field.len(), values.len());
        for (i, &v) in values.iter().enumerate() {
            prop_assert_eq!(field.values()[i].to_bits(), v.to_bits(),
                "bin {} not preserved", i);
        }
        prop_assert!(field.metadata().creation_hash().is_some());
    }
}
