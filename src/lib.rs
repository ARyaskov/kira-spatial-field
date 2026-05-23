#![deny(unsafe_code)]

//! Deterministic scalar field data structures for spatial domains.

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

pub use axis::{AxisDefinition, AxisDefinitionBuilder};
#[cfg(feature = "field-cache")]
pub use cache::{FieldCache, cached_or_compute};
pub use error::{FieldError, FieldErrorExt};
pub use field::Field;
pub use gene_field::axis_expressing_mask;
pub use metadata::{FieldMetadata, FieldMetadataBuilder};
pub use normalization::NormalizationFlags;
pub use reduction::{PanelReduction, ReductionMethod};
