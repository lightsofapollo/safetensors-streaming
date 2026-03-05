use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// Re-export DType from sst-types so existing consumers don't break
pub use sst_types::DType;

/// Metadata for a single tensor in a safetensors file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Byte offsets (start, end) relative to the data section start.
    pub data_offsets: (usize, usize),
}

impl TensorInfo {
    /// Total size of this tensor's data in bytes (product of shape dims * dtype byte size).
    pub fn byte_size(&self) -> usize {
        self.shape.iter().copied().product::<usize>() * self.dtype.byte_size()
    }

    /// Convert relative data offsets to absolute file offsets.
    pub fn absolute_offsets(&self, data_start: usize) -> (usize, usize) {
        (data_start + self.data_offsets.0, data_start + self.data_offsets.1)
    }
}

/// Parsed safetensors header.
#[derive(Debug, Clone)]
pub struct Header {
    /// Size of the JSON header in bytes (the u64 from bytes 0..8).
    pub header_size: u64,
    /// Offset where the data section begins (8 + header_size).
    pub data_start: usize,
    /// Tensors ordered by data_offsets.0 for sequential access.
    pub tensors: Vec<TensorInfo>,
    /// Optional metadata from the "__metadata__" key.
    pub metadata: HashMap<String, String>,
}
