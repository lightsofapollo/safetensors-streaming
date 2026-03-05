/// Errors from GPU operations.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("CUDA not available")]
    CudaNotAvailable,

    #[error("device {0} not found")]
    DeviceNotFound(u32),

    #[error("buffer full: requested {requested} bytes but only {available} available")]
    BufferFull { requested: usize, available: usize },

    #[error("invalid slot: offset {offset} len {len} exceeds buffer capacity {capacity}")]
    InvalidSlot {
        offset: usize,
        len: usize,
        capacity: usize,
    },

    #[error("allocation size {0} is zero")]
    ZeroAllocation(usize),

    #[cfg(feature = "cuda")]
    #[error("CUDA driver error: {0}")]
    CudaDriver(#[from] cudarc::driver::DriverError),

    #[error("failed to initialize CUDA device {ordinal}: {reason}")]
    DeviceInit { ordinal: usize, reason: String },

    #[error("failed to create CUDA stream: {0}")]
    StreamCreate(String),

    #[error("device memory allocation failed ({bytes} bytes): {reason}")]
    Alloc { bytes: usize, reason: String },

    #[error("host-to-device copy failed ({bytes} bytes): {reason}")]
    CopyH2D { bytes: usize, reason: String },

    #[error("stream synchronize failed: {0}")]
    Synchronize(String),

    #[error("pinned buffer allocation failed ({bytes} bytes): {reason}")]
    PinnedAlloc { bytes: usize, reason: String },

    #[error("pinned buffer staging failed ({bytes} bytes): {reason}")]
    PinnedStage { bytes: usize, reason: String },
}
