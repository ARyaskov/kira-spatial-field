//! `kira-spatial-field` defines deterministic scalar field data structures for spatial domains.
//!
//! This crate provides structural contracts for field values and metadata only.
//! It intentionally excludes spatial operators, sparse/CSR math, and transformation logic.
//! Computation and domain primitives remain separated in `kira-spatial-core`.
//! Optional SIMD acceleration is available as a performance layer only and
//! never changes deterministic scalar semantics; scalar fallback is always present.

pub mod axis;
#[cfg(feature = "field-cache")]
pub mod cache;
pub mod error;
pub mod field;
pub mod gene_field;
pub mod metadata;
pub mod normalization;
pub mod reduction;
mod simd;

#[cfg(feature = "field-cache")]
pub use cache::{FieldCache, cached_or_compute};
pub use field::Field;
