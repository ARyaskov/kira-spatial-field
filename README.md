# kira-spatial-field

`kira-spatial-field` is a deterministic Rust library for constructing immutable scalar fields from spatial transcriptomics expression matrices.

## Purpose

- Deterministic gene-field, panel-field, and axis-field construction over a spatial domain.
- Reproducible scalar reductions with explicit normalization only.
- Immutable field container with provenance-oriented metadata and creation hashes.

## Core Capabilities

- Single-gene extraction via `Field::from_gene`.
- Multi-gene panel construction via `Field::from_panel`.
- Reproducible biological axis construction via `Field::from_axis`.
- Explicit pure transforms: `log1p`, `zscore_global`, `zscore_masked`, and `minmax_scale`.
- Stable `AxisDefinition` hashing and optional in-memory field caching.

## Determinism Guarantees

- Fixed spatial bin iteration order with no hidden reordering.
- Fixed gene iteration order; panel inputs must already be lexicographically sorted.
- No implicit normalization during field construction.
- No parallel floating-point reduction trees.
- Stable SHA-256 `definition_hash` for `AxisDefinition`.
- Stable SHA-256 `creation_hash` for fully constructed `Field` values.
- Bitwise-identical output for identical inputs and feature configuration.

## Immutability Model

- `Field` stores values in `Arc<[f32]>` and exposes read-only slices only.
- `FieldMetadata` is immutable outside the crate and exposed via getters only.
- All transforms allocate a new field; original fields remain unchanged.

## Compatibility

- **MSRV:** Rust 1.95 (Edition 2024). The crate uses let-chains and other
  recent stable surface; older toolchains will refuse to compile.
- The crate uses `#![deny(unsafe_code)]`. The only modules that opt back
  in (via `#[allow(unsafe_code)]`) are the gated `simd::avx2` /
  `simd::neon` intrinsics. Every other module is unsafe-free.
- Public enums are `#[non_exhaustive]` — downstream `match` arms must
  include a `_ => …` fallback so future variants stay non-breaking.
- All public numeric entry points reject `NaN`/`Inf`; the SHA-256
  `creation_hash` is the same across runs for the same input.

## Feature Flags

| Feature | Default | Purpose |
|---|---:|---|
| `simd` | no | Enables optional per-element SIMD acceleration for safe transform passes without changing results. |
| `field-cache` | no | Enables optional bounded in-memory caching keyed by `creation_hash`. |

## Cache Behavior

When `field-cache` is enabled:

- `FieldCache` stores fully constructed immutable `Arc<Field>` values.
- Cache key is `creation_hash` only.
- Cache use is explicit through `cached_or_compute`.
- Cache never changes field contents or output semantics; it only avoids recomputation.

## SIMD Behavior

When `simd` is enabled:

- SIMD is used only for eligible per-element transforms.
- Reductions, sorting, variance accumulation, and weighted accumulation remain scalar.
- Scalar fallback is always available.
- SIMD must preserve bitwise equality with the scalar path.

## Versioning Policy

- Crate version follows semver (`0.x.y` during early development).
- `ReductionMethod` discriminants and hash byte order are part of the public determinism contract.
- Breaking hash semantics or persisted metadata compatibility requires a major version bump.

## Examples

A runnable end-to-end example lives in [`examples/axis_field.rs`](examples/axis_field.rs):

```bash
cargo run --example axis_field
```

## Minimal Usage

```rust
use std::collections::HashMap;

use kira_spatial_field::{
    Field,
    axis::AxisDefinition,
    gene_field::{ExpressionCsrView, SpatialDomainView},
    normalization::NormalizationFlags,
    reduction::ReductionMethod,
};

struct MockCsr {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f32>,
    gene_index: HashMap<String, usize>,
}

impl ExpressionCsrView for MockCsr {
    fn indptr(&self) -> &[usize] { &self.indptr }
    fn indices(&self) -> &[usize] { &self.indices }
    fn data(&self) -> &[f32] { &self.data }
    fn gene_index(&self) -> &HashMap<String, usize> { &self.gene_index }
}

struct MockDomain {
    id: u64,
    bin_count: usize,
}

impl SpatialDomainView for MockDomain {
    fn id(&self) -> u64 { self.id }
    fn bin_count(&self) -> usize { self.bin_count }
}

fn build_axis_field() -> Result<Field, kira_spatial_field::FieldError> {
    let mut gene_index = HashMap::new();
    gene_index.insert("G1".to_string(), 0);
    gene_index.insert("G2".to_string(), 1);

    let csr = MockCsr {
        indptr: vec![0, 2, 3],
        indices: vec![0, 1, 1],
        data: vec![1.0, 3.0, 2.0],
        gene_index,
    };

    let domain = MockDomain { id: 42, bin_count: 2 };

    let axis = AxisDefinition::new(
        "immune_axis".to_string(),
        1,
        vec!["G1".to_string(), "G2".to_string()],
        None,
        ReductionMethod::Mean,
        NormalizationFlags::default(),
    )?;

    Field::from_axis(&axis, &csr, &domain)
}
```
