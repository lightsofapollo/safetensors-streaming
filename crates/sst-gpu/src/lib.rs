// sst-gpu: CUDA copy management (stub)
//
// Modules (planned):
// - device: CUDA device discovery and selection
// - copy: Async H2D copy with pinned staging buffers
// - stream: CUDA stream management for overlapping transfers

pub mod error;

pub use error::GpuError;

/// Placeholder for GPU device handle.
/// Will wrap CUDA device context when CUDA support is added.
pub struct GpuDevice {
    device_id: u32,
}

impl GpuDevice {
    /// Create a stub GPU device reference.
    pub fn stub(device_id: u32) -> Self {
        tracing::info!(device_id, "created stub GPU device");
        Self { device_id }
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}
