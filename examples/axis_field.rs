//! Minimal end-to-end example: build an axis-driven field from a tiny
//! synthetic CSR.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example axis_field
//! ```

use std::collections::HashMap;

use kira_spatial_field::{
    AxisDefinition, Field, FieldError, NormalizationFlags, ReductionMethod,
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

fn build_axis_field() -> Result<Field, FieldError> {
    let mut gene_index = HashMap::new();
    gene_index.insert("G1".to_string(), 0);
    gene_index.insert("G2".to_string(), 1);

    let csr = MockCsr {
        indptr: vec![0, 2, 3],
        indices: vec![0, 1, 1],
        data: vec![1.0, 3.0, 2.0],
        gene_index,
    };

    let domain = MockDomain {
        id: 42,
        bin_count: 2,
    };

    let axis = AxisDefinition::builder(
        "immune_axis",
        1,
        vec!["G1".to_string(), "G2".to_string()],
        ReductionMethod::Mean,
    )
    .with_normalization(NormalizationFlags::default())
    .build()?;

    Field::from_axis(&axis, &csr, &domain)
}

fn main() {
    let field = build_axis_field().expect("axis field should construct");
    println!("{}", field);
    println!("values: {:?}", field.values());
    println!("creation_hash: {:?}", field.metadata().creation_hash().map(hex::encode_short));
}

mod hex {
    pub fn encode_short(bytes: [u8; 32]) -> String {
        let mut s = String::with_capacity(16);
        for b in &bytes[..8] {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }
}
