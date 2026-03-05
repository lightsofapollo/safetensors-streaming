// sst-types: Shared types used across sst crates (no internal crate dependencies)

use serde::{Deserialize, Serialize};

/// Supported tensor data types in safetensors format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    Bool,
    U8,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

/// Error returned when parsing an unknown dtype string.
#[derive(Debug, thiserror::Error)]
#[error("unknown dtype: {0:?}")]
pub struct UnknownDTypeError(pub String);

impl std::str::FromStr for DType {
    type Err = UnknownDTypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "BOOL" => Ok(Self::Bool),
            "U8" => Ok(Self::U8),
            "I8" => Ok(Self::I8),
            "I16" => Ok(Self::I16),
            "I32" => Ok(Self::I32),
            "I64" => Ok(Self::I64),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "F32" => Ok(Self::F32),
            "F64" => Ok(Self::F64),
            other => Err(UnknownDTypeError(other.to_string())),
        }
    }
}

impl DType {
    /// Size of a single element in bytes.
    pub fn byte_size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
}
