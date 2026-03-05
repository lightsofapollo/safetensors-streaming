// sst-gpu/src/pipeline.rs — Full streaming pipeline: Consumer -> pinned staging -> GPU
//
// All CUDA code is behind `#[cfg(feature = "cuda")]`.
// Depends on:
//   - sst-buffer (Consumer, TensorChunk)
//   - sst-types (DType)
//   - crate::copy (GpuCopier, DeviceBuffer)
//   - crate::pinned (PinnedBuffer)

use sst_buffer::Consumer;
use sst_types::DType;

use crate::copy::GpuCopier;
use crate::pinned::PinnedBuffer;
use crate::GpuError;

/// A tensor whose data now lives in GPU device memory.
#[derive(Debug)]
pub struct GpuTensor {
    /// Tensor name from the safetensors file.
    pub name: String,
    /// Raw device pointer (suitable for wrapping via PyTorch `from_blob`).
    pub device_ptr: u64,
    /// Length in bytes on device.
    pub len: usize,
    /// Element data type.
    pub dtype: DType,
    /// Tensor shape (dimensions).
    pub shape: Vec<usize>,
    // Hold the device buffer alive so the allocation isn't freed.
    _device_buf: crate::copy::DeviceBuffer,
}

/// Orchestrates the full streaming pipeline:
///
/// `Consumer` -> stage in pinned host memory -> async H2D copy -> [`GpuTensor`]
pub struct GpuPipeline {
    copier: GpuCopier,
    pinned: PinnedBuffer,
}

impl GpuPipeline {
    /// Create a pipeline targeting `device_ordinal` with a pinned staging
    /// buffer of `pinned_buffer_size` bytes.
    pub fn new(
        device_ordinal: usize,
        pinned_buffer_size: usize,
    ) -> Result<Self, GpuError> {
        let copier = GpuCopier::new(device_ordinal)?;
        let pinned = PinnedBuffer::new(copier.device(), pinned_buffer_size).map_err(|e| {
            tracing::error!(size = pinned_buffer_size, err = %e, "failed to allocate pinned buffer");
            GpuError::PinnedAlloc {
                bytes: pinned_buffer_size,
                reason: e.to_string(),
            }
        })?;

        tracing::info!(
            device_ordinal,
            pinned_buffer_size,
            "GpuPipeline initialized"
        );
        Ok(Self { copier, pinned })
    }

    /// Drain all tensor chunks from `consumer`, copy each to the GPU, and
    /// return the resulting [`GpuTensor`] handles.
    ///
    /// For each chunk the flow is:
    /// 1. Stage data into pinned host memory via `write_slot`.
    /// 2. Launch async H2D copy on the copier's stream.
    /// 3. Synchronize (ensures copy is complete before reusing the pinned slot).
    /// 4. Release the pinned slot.
    /// 5. Record the resulting `GpuTensor`.
    pub async fn process(
        &mut self,
        mut consumer: Consumer,
    ) -> Result<Vec<GpuTensor>, GpuError> {
        let mut tensors: Vec<GpuTensor> = Vec::new();

        while let Some(chunk) = consumer.recv().await {
            let data = &chunk.data;
            let len = data.len();

            tracing::debug!(name = %chunk.name, bytes = len, "staging tensor");

            // Stage into pinned memory.
            let slot = self.pinned.write_slot(data).map_err(|e| {
                tracing::error!(name = %chunk.name, err = %e, "pinned staging failed");
                GpuError::PinnedStage {
                    bytes: len,
                    reason: e.to_string(),
                }
            })?;

            // Async copy from pinned host -> device.
            let device_buf = self.copier.copy_to_device(slot.ptr, len)?;

            // Synchronize to ensure copy is done before we reuse the pinned slot.
            self.copier.synchronize()?;

            // Release the pinned staging slot so it can be reused.
            self.pinned.release_slot(slot);

            let gpu_tensor = GpuTensor {
                name: chunk.name,
                device_ptr: device_buf.device_ptr(),
                len,
                dtype: chunk.dtype,
                shape: chunk.shape,
                _device_buf: device_buf,
            };

            tracing::debug!(
                name = %gpu_tensor.name,
                device_ptr = gpu_tensor.device_ptr,
                bytes = gpu_tensor.len,
                "tensor on device"
            );

            tensors.push(gpu_tensor);
        }

        tracing::info!(count = tensors.len(), "pipeline complete");
        Ok(tensors)
    }
}
