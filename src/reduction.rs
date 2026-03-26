use serde::{Deserialize, Serialize};

/// Deterministic scalar-reduction strategy over selected genes.
///
/// The ordering of `source_genes` in metadata must be deterministic and stable.
/// Stages implementing these methods must preserve that order and avoid
/// non-deterministic floating-point accumulation behavior.
///
/// The discriminant mapping is part of the hash contract and must remain stable.
/// Reordering variants or changing discriminants requires a major version bump
/// for consumers relying on persisted hashes.
#[repr(u8)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReductionMethod {
    SingleGene = 0,
    Mean = 1,
    TrimmedMean { trim_fraction: f32 } = 2,
    Weighted = 3,
}

impl ReductionMethod {
    /// Returns the explicit stable discriminant used by deterministic hashing.
    pub const fn discriminant(self: &Self) -> u8 {
        match self {
            Self::SingleGene => 0,
            Self::Mean => 1,
            Self::TrimmedMean { .. } => 2,
            Self::Weighted => 3,
        }
    }

    /// Returns extra deterministic bytes for configuration-bearing variants.
    pub fn hash_payload(&self) -> Option<[u8; 4]> {
        match self {
            Self::TrimmedMean { trim_fraction } => Some(trim_fraction.to_le_bytes()),
            _ => None,
        }
    }
}
