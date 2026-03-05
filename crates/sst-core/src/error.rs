/// Errors from core safetensors parsing and planning.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("header too short: got {got} bytes, expected at least {expected}")]
    HeaderTooShort { got: usize, expected: usize },

    #[error("invalid JSON in header: {0}")]
    InvalidJson(#[from] serde_json::Error),

    #[error("header is not a JSON object")]
    InvalidHeaderStructure,

    #[error("tensor {tensor:?} missing field {field:?}")]
    MissingField { tensor: String, field: String },

    #[error("unknown dtype: {0}")]
    UnknownDType(#[from] sst_types::UnknownDTypeError),

    #[error("fetch error: {0}")]
    Fetch(#[from] sst_fetch::FetchError),

    #[error("buffer error: {0}")]
    Buffer(#[from] sst_buffer::BufferError),
}
