use thiserror::Error;

/// Error surface for field construction and validation stages.
#[derive(Debug, Error)]
pub enum FieldError {
    #[error("field values length does not match the bound domain size")]
    DomainSizeMismatch,
    #[error("field values contain invalid floating-point entries")]
    InvalidValues,
    #[error("reduction method configuration is invalid")]
    InvalidReduction,
    #[error("field metadata is invalid")]
    InvalidMetadata,
}
