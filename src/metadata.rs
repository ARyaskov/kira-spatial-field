use crate::{normalization::NormalizationFlags, reduction::ReductionMethod};
use serde::{Deserialize, Serialize};

/// Metadata describing how a [`crate::field::Field`] is defined.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMetadata {
    field_name: String,
    source_genes: Vec<String>,
    reduction_method: ReductionMethod,
    normalization_flags: NormalizationFlags,
    creation_hash: Option<[u8; 32]>,
    axis_hash: Option<[u8; 32]>,
}

impl FieldMetadata {
    pub(crate) fn new(
        field_name: String,
        source_genes: Vec<String>,
        reduction_method: ReductionMethod,
        normalization_flags: NormalizationFlags,
        axis_hash: Option<[u8; 32]>,
    ) -> Self {
        Self {
            field_name,
            source_genes,
            reduction_method,
            normalization_flags,
            creation_hash: None,
            axis_hash,
        }
    }

    /// Start a builder.
    pub fn builder(
        field_name: impl Into<String>,
        source_genes: Vec<String>,
        reduction_method: ReductionMethod,
    ) -> FieldMetadataBuilder {
        FieldMetadataBuilder {
            field_name: field_name.into(),
            source_genes,
            reduction_method,
            normalization_flags: NormalizationFlags::default(),
            axis_hash: None,
        }
    }

    pub(crate) fn append_field_name_suffix(&mut self, suffix: &str) {
        self.field_name.push_str(suffix);
    }

    pub(crate) fn set_normalization_flags(&mut self, normalization_flags: NormalizationFlags) {
        self.normalization_flags = normalization_flags;
    }

    pub(crate) fn set_creation_hash(&mut self, creation_hash: [u8; 32]) {
        self.creation_hash = Some(creation_hash);
    }

    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    pub fn source_genes(&self) -> &[String] {
        &self.source_genes
    }

    pub fn reduction_method(&self) -> &ReductionMethod {
        &self.reduction_method
    }

    pub fn normalization_flags(&self) -> &NormalizationFlags {
        &self.normalization_flags
    }

    pub fn creation_hash(&self) -> Option<[u8; 32]> {
        self.creation_hash
    }

    pub fn axis_hash(&self) -> Option<[u8; 32]> {
        self.axis_hash
    }
}

pub struct FieldMetadataBuilder {
    field_name: String,
    source_genes: Vec<String>,
    reduction_method: ReductionMethod,
    normalization_flags: NormalizationFlags,
    axis_hash: Option<[u8; 32]>,
}

impl FieldMetadataBuilder {
    pub fn with_normalization_flags(mut self, flags: NormalizationFlags) -> Self {
        self.normalization_flags = flags;
        self
    }
    pub fn with_axis_hash(mut self, hash: [u8; 32]) -> Self {
        self.axis_hash = Some(hash);
        self
    }
    pub fn build(self) -> FieldMetadata {
        FieldMetadata::new(
            self.field_name,
            self.source_genes,
            self.reduction_method,
            self.normalization_flags,
            self.axis_hash,
        )
    }
}
