// sst-core: Header parsing, fetch planning, and pipeline orchestration

pub mod error;
pub mod header;
pub mod pipeline;
pub mod types;

pub use error::CoreError;
pub use header::{parse_header, parse_header_json, parse_header_size};
pub use pipeline::{PipelineConfig, StreamingPipeline};
pub use types::{DType, Header, TensorInfo};
