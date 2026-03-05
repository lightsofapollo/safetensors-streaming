// sst-gpu: CUDA copy management
//
// Modules:
// - pinned: Ring buffer (HeapBuffer always, PinnedBuffer behind cuda feature)
// - copy: Async H2D copy with pinned staging buffers (cuda only)
// - pipeline: Full streaming pipeline Consumer -> pinned -> GPU (cuda only)

pub mod error;
pub mod pinned;

#[cfg(feature = "cuda")]
pub mod copy;
#[cfg(feature = "cuda")]
pub mod pipeline;

pub use error::GpuError;
pub use pinned::{HeapBuffer, HeapSlot};

#[cfg(feature = "cuda")]
pub use pinned::{PinnedBuffer, PinnedSlot};
#[cfg(feature = "cuda")]
pub use copy::{DeviceBuffer, GpuCopier};
#[cfg(feature = "cuda")]
pub use pipeline::{GpuPipeline, GpuTensor};

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
