// sst-gpu/src/copy.rs — Async GPU copy via dedicated CUDA stream
//
// All CUDA code is behind `#[cfg(feature = "cuda")]` (this file is only
// compiled when the cuda feature is active, gated in lib.rs).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};

use crate::GpuError;

/// Handle to device memory allocated by [`GpuCopier`].
///
/// Wraps a `CudaSlice<u8>` so callers can extract the raw device pointer
/// (e.g. to hand to PyTorch via `from_blob`).
pub struct DeviceBuffer {
    inner: CudaSlice<u8>,
    stream: Arc<CudaStream>,
}

impl DeviceBuffer {
    /// Raw device pointer as `u64`, suitable for FFI / PyTorch interop.
    pub fn device_ptr(&self) -> u64 {
        let (ptr, _sync) = self.inner.device_ptr(&self.stream);
        ptr as u64
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the buffer is empty (zero-length).
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Consume and return the underlying `CudaSlice` for direct use with cudarc.
    pub fn into_inner(self) -> CudaSlice<u8> {
        self.inner
    }
}

/// Manages a dedicated CUDA stream for async host-to-device copies.
///
/// Typical flow:
/// 1. Create a `GpuCopier` bound to a device.
/// 2. Call [`copy_to_device`] one or more times -- each launches an async memcpy.
/// 3. Call [`synchronize`] when you need results to be ready.
pub struct GpuCopier {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl GpuCopier {
    /// Create a new copier with its own CUDA stream on `device_ordinal`.
    pub fn new(device_ordinal: usize) -> Result<Self, GpuError> {
        let ctx = CudaContext::new(device_ordinal).map_err(|e| {
            tracing::error!(device_ordinal, err = %e, "failed to acquire CUDA device");
            GpuError::DeviceInit {
                ordinal: device_ordinal,
                reason: e.to_string(),
            }
        })?;

        let stream = ctx.new_stream().map_err(|e| {
            tracing::error!(err = %e, "failed to create CUDA stream");
            GpuError::StreamCreate(e.to_string())
        })?;

        tracing::debug!(device_ordinal, "GpuCopier ready");
        Ok(Self { ctx, stream })
    }

    /// Asynchronously copy `len` bytes from `host_ptr` to device memory.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `host_ptr` points to at least `len` bytes of valid, **pinned** host memory.
    /// - The memory remains valid until [`synchronize`] completes.
    ///
    /// Using non-pinned memory will still work but CUDA will silently
    /// fall back to a synchronous copy, defeating the purpose.
    pub fn copy_to_device(
        &self,
        host_ptr: *const u8,
        len: usize,
    ) -> Result<DeviceBuffer, GpuError> {
        if len == 0 {
            let empty: CudaSlice<u8> = self.stream.alloc_zeros(0).map_err(|e| {
                GpuError::Alloc {
                    bytes: 0,
                    reason: e.to_string(),
                }
            })?;
            return Ok(DeviceBuffer { inner: empty, stream: Arc::clone(&self.stream) });
        }

        // Build a host slice from the raw pointer (no ownership taken).
        // SAFETY: caller guarantees validity for `len` bytes.
        let host_slice = unsafe { std::slice::from_raw_parts(host_ptr, len) };

        // Copy host data to device via the stream.
        let device_buf: CudaSlice<u8> =
            self.stream.memcpy_stod(host_slice).map_err(|e| {
                tracing::error!(bytes = len, err = %e, "H2D copy failed");
                GpuError::CopyH2D {
                    bytes: len,
                    reason: e.to_string(),
                }
            })?;

        tracing::trace!(bytes = len, "queued H2D copy");
        Ok(DeviceBuffer { inner: device_buf, stream: Arc::clone(&self.stream) })
    }

    /// Block until all work on this copier's stream has completed.
    pub fn synchronize(&self) -> Result<(), GpuError> {
        self.stream.synchronize().map_err(|e| {
            tracing::error!(err = %e, "stream synchronize failed");
            GpuError::Synchronize(e.to_string())
        })?;
        tracing::trace!("stream synchronized");
        Ok(())
    }

    /// Reference to the underlying `CudaContext`.
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}
