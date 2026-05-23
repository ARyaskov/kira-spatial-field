#![cfg(feature = "simd")]

use std::collections::HashMap;

use kira_spatial_field::{
    Field,
    gene_field::{ExpressionCsrView, SpatialDomainView},
};

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

fn bitwise_eq(lhs: &[f32], rhs: &[f32]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn make_field(values: &[f32]) -> Field {
    let mut gene_index = HashMap::new();
    gene_index.insert("G0".to_string(), 0);

    let mut indptr: Vec<u32> = Vec::with_capacity(values.len() + 1);
    indptr.push(0);
    for i in 0..values.len() {
        indptr.push((i + 1) as u32);
    }

    let csr = MockCsr {
        indptr,
        indices: vec![0_u32; values.len()],
        data: values.to_vec(),
        gene_index,
    };
    let domain = MockDomain {
        id: 7,
        bin_count: values.len(),
    };

    Field::from_gene(&csr, "G0", &domain).expect("field should construct")
}

#[test]
fn simd_enabled_transforms_match_scalar_formulas_bitwise() {
    let source = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 21.0, 50.0];
    let field = make_field(&source);

    let log1p = field.log1p().expect("log1p should succeed");
    let expected_log1p: Vec<f32> = source.iter().map(|v| (1.0_f32 + *v).ln()).collect();
    assert!(bitwise_eq(log1p.values(), &expected_log1p));

    // zscore_global uses f64 Welford internally (see field.rs); the
    // reference mirrors that — accumulate in f64, downcast to f32 only
    // at the boundary that feeds `apply_sub_div`.
    let zscore = field.zscore_global().expect("zscore should succeed");
    let mut count: u64 = 0;
    let mut mean64 = 0.0_f64;
    let mut m2 = 0.0_f64;
    for &v in &source {
        let x = v as f64;
        count += 1;
        let delta = x - mean64;
        mean64 += delta / count as f64;
        let delta2 = x - mean64;
        m2 += delta * delta2;
    }
    let mean = mean64 as f32;
    let sigma = ((m2 / count as f64).sqrt()) as f32;
    let expected_z: Vec<f32> = if sigma == 0.0 {
        vec![0.0; source.len()]
    } else {
        source.iter().map(|v| (*v - mean) / sigma).collect()
    };
    assert!(bitwise_eq(zscore.values(), &expected_z));

    let minmax = field.minmax_scale().expect("minmax should succeed");
    let min = source.iter().copied().fold(f32::INFINITY, f32::min);
    let max = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let expected_minmax: Vec<f32> = if max == min {
        vec![0.0; source.len()]
    } else {
        source.iter().map(|v| (*v - min) / (max - min)).collect()
    };
    assert!(bitwise_eq(minmax.values(), &expected_minmax));
}
