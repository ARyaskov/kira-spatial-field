use thiserror::Error;

/// Error surface for field construction and validation.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FieldError {
    #[error("field values length does not match the bound domain size")]
    DomainSizeMismatch,
    #[error("field values contain invalid floating-point entries")]
    InvalidValues,
    #[error("reduction method configuration is invalid")]
    InvalidReduction,
    #[error("field metadata is invalid")]
    InvalidMetadata,
    #[error("gene not found in expression matrix: {gene_id}")]
    GeneNotFound { gene_id: String },
    #[error("{what}: {source}")]
    WithContext {
        what: &'static str,
        #[source]
        source: Box<FieldError>,
    },
}

impl FieldError {
    pub fn context(self, what: &'static str) -> Self {
        Self::WithContext {
            what,
            source: Box::new(self),
        }
    }
}

pub trait FieldErrorExt<T> {
    fn with_context(self, what: &'static str) -> Result<T, FieldError>;
}

impl<T> FieldErrorExt<T> for Result<T, FieldError> {
    fn with_context(self, what: &'static str) -> Result<T, FieldError> {
        self.map_err(|e| e.context(what))
    }
}
