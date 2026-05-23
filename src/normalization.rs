use serde::{Deserialize, Serialize};

/// Declared normalization operations associated with a field definition.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct NormalizationFlags {
    pub log1p: bool,
    pub zscore_global: bool,
    pub zscore_masked: bool,
    pub minmax_scale: bool,
}
