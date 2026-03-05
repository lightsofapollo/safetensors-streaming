/// Errors from GPU operations.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("CUDA not available")]
    CudaNotAvailable,

    #[error("device {0} not found")]
    DeviceNotFound(u32),
}
