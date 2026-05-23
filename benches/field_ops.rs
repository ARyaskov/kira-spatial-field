//! Micro-benchmarks for the hot paths.
//!
//! Run with:
//!
//! ```bash
//! cargo bench --bench field_ops
//! cargo bench --bench field_ops --features simd
//! ```

use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use kira_spatial_field::{
    Field,
    gene_field::{ExpressionCsrView, SpatialDomainView},
    reduction::ReductionMethod,
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

/// Synthesise a CSR with `bins` rows and `genes` columns. Each row has
/// `nnz_per_row` non-zero entries spread evenly across the gene axis.
fn build_csr(bins: usize, genes: usize, nnz_per_row: usize) -> (MockCsr, Vec<String>) {
    let gene_ids: Vec<String> = (0..genes).map(|i| format!("G{:05}", i)).collect();
    let gene_index: HashMap<String, usize> = gene_ids
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();

    let mut indptr: Vec<u32> = Vec::with_capacity(bins + 1);
    indptr.push(0_u32);
    let mut indices: Vec<u32> = Vec::with_capacity(bins * nnz_per_row);
    let mut data = Vec::with_capacity(bins * nnz_per_row);
    for b in 0..bins {
        for k in 0..nnz_per_row {
            // Evenly spaced columns within `genes`, shifted by row index
            // for variation. CSR contract: per-row indices sorted asc.
            let col = ((b + k) * genes / nnz_per_row.max(1)) % genes;
            indices.push(col as u32);
            data.push((b * nnz_per_row + k) as f32 * 0.001);
        }
        // Re-sort row indices to satisfy CSR contract (small array).
        let start = *indptr.last().unwrap() as usize;
        let end = start + nnz_per_row;
        let pair_slice: Vec<(u32, f32)> = (start..end).map(|i| (indices[i], data[i])).collect();
        let mut pair_sorted = pair_slice;
        pair_sorted.sort_unstable_by_key(|&(c, _)| c);
        pair_sorted.dedup_by_key(|(c, _)| *c);
        let new_end = start + pair_sorted.len();
        for (i, (c, d)) in pair_sorted.into_iter().enumerate() {
            indices[start + i] = c;
            data[start + i] = d;
        }
        indices.truncate(new_end);
        data.truncate(new_end);
        indptr.push(new_end as u32);
    }

    (
        MockCsr {
            indptr,
            indices,
            data,
            gene_index,
        },
        gene_ids,
    )
}

fn bench_from_gene(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_gene");
    for &(bins, genes, nnz) in &[(1024, 2000, 50), (8192, 2000, 50)] {
        let (csr, gene_ids) = build_csr(bins, genes, nnz);
        let domain = MockDomain {
            id: 1,
            bin_count: bins,
        };
        group.throughput(Throughput::Elements(bins as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", bins, genes)),
            &(),
            |b, _| {
                b.iter(|| {
                    let f = Field::from_gene(&csr, &gene_ids[100], &domain).unwrap();
                    criterion::black_box(f);
                })
            },
        );
    }
    group.finish();
}

fn bench_from_panel(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_panel_mean");
    for &(bins, genes, nnz, panel_size) in &[
        (1024, 2000, 50, 30),
        (8192, 2000, 50, 30),
        (8192, 2000, 50, 100),
    ] {
        let (csr, gene_ids) = build_csr(bins, genes, nnz);
        let domain = MockDomain {
            id: 1,
            bin_count: bins,
        };
        let mut panel_ids: Vec<String> = gene_ids.iter().take(panel_size).cloned().collect();
        panel_ids.sort_unstable();
        group.throughput(Throughput::Elements((bins * panel_size) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}/panel{}", bins, genes, panel_size)),
            &(),
            |b, _| {
                b.iter(|| {
                    let f =
                        Field::from_panel(&csr, &panel_ids, ReductionMethod::Mean, None, &domain)
                            .unwrap();
                    criterion::black_box(f);
                })
            },
        );
    }
    group.finish();
}

fn bench_zscore_global(c: &mut Criterion) {
    let mut group = c.benchmark_group("zscore_global");
    for &bins in &[1024_usize, 8192, 65536] {
        let (csr, gene_ids) = build_csr(bins, 100, 50);
        let domain = MockDomain {
            id: 1,
            bin_count: bins,
        };
        let field = Field::from_gene(&csr, &gene_ids[10], &domain).unwrap();
        group.throughput(Throughput::Elements(bins as u64));
        group.bench_with_input(BenchmarkId::from_parameter(bins), &(), |b, _| {
            b.iter(|| {
                let z = field.zscore_global().unwrap();
                criterion::black_box(z);
            })
        });
    }
    group.finish();
}

criterion_group!(field_benches, bench_from_gene, bench_from_panel, bench_zscore_global);
criterion_main!(field_benches);
