use crate::error::FieldError;
use serde::{Deserialize, Serialize};

/// Deterministic scalar-reduction strategy over selected genes.
/// Discriminants are part of the hash contract — do not reorder.
#[repr(u8)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ReductionMethod {
    SingleGene = 0,
    Mean = 1,
    TrimmedMean { trim_fraction: f32 } = 2,
    Weighted = 3,
}

impl ReductionMethod {
    pub const fn discriminant(&self) -> u8 {
        match self {
            Self::SingleGene => 0,
            Self::Mean => 1,
            Self::TrimmedMean { .. } => 2,
            Self::Weighted => 3,
        }
    }

    pub fn hash_payload(&self) -> Option<[u8; 4]> {
        match self {
            Self::TrimmedMean { trim_fraction } => Some(trim_fraction.to_le_bytes()),
            _ => None,
        }
    }
}

/// Reduction strategy restricted to panel (multi-gene) modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PanelReduction {
    Mean,
    TrimmedMean { trim_fraction: f32 },
    Weighted,
}

impl From<PanelReduction> for ReductionMethod {
    fn from(value: PanelReduction) -> Self {
        match value {
            PanelReduction::Mean => ReductionMethod::Mean,
            PanelReduction::TrimmedMean { trim_fraction } => {
                ReductionMethod::TrimmedMean { trim_fraction }
            }
            PanelReduction::Weighted => ReductionMethod::Weighted,
        }
    }
}

impl TryFrom<ReductionMethod> for PanelReduction {
    type Error = FieldError;
    fn try_from(value: ReductionMethod) -> Result<Self, Self::Error> {
        match value {
            ReductionMethod::Mean => Ok(PanelReduction::Mean),
            ReductionMethod::TrimmedMean { trim_fraction } => {
                Ok(PanelReduction::TrimmedMean { trim_fraction })
            }
            ReductionMethod::Weighted => Ok(PanelReduction::Weighted),
            ReductionMethod::SingleGene => Err(FieldError::InvalidReduction),
        }
    }
}
