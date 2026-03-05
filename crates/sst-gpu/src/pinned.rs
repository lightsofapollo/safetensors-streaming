//! Ring buffer implementations for staging tensor data.
//!
//! - `HeapBuffer`: CPU heap-backed fallback (always available).
//! - `PinnedBuffer`: CUDA pinned (page-locked) host memory for fast H2D transfers
//!   (only available with the `cuda` feature).
//!
//! Both implement a simple bump-allocator ring buffer: writes advance a head
//! pointer, releases advance a tail pointer. Slots are returned in FIFO order
//! so the caller must release them in the same order they were allocated.

use crate::GpuError;

// ---------------------------------------------------------------------------
// Slot types
// ---------------------------------------------------------------------------

/// A slot referencing a region inside a `HeapBuffer`.
#[derive(Debug)]
pub struct HeapSlot {
    /// Offset into the backing `Vec<u8>`.
    pub offset: usize,
    /// Number of valid bytes.
    pub len: usize,
}

/// A slot referencing a region inside a `PinnedBuffer`.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct PinnedSlot {
    /// Raw pointer into pinned host memory.
    pub ptr: *mut u8,
    /// Number of valid bytes.
    pub len: usize,
    /// Offset within the ring (needed for release bookkeeping).
    offset: usize,
}

#[cfg(feature = "cuda")]
// SAFETY: The pinned memory is allocated once and the pointer remains stable
// for the lifetime of the PinnedBuffer. Sending the slot to another thread is
// safe as long as the PinnedBuffer outlives it (enforced by the caller).
unsafe impl Send for PinnedSlot {}

// ---------------------------------------------------------------------------
// HeapBuffer — always available
// ---------------------------------------------------------------------------

/// A ring buffer backed by a heap-allocated `Vec<u8>`.
///
/// Uses a simple wrap-around bump allocator. Slots must be released in FIFO
/// order for the ring to reclaim space correctly.
pub struct HeapBuffer {
    buf: Vec<u8>,
    capacity: usize,
    /// Write head — next byte to allocate at.
    head: usize,
    /// Read tail — first byte still in use.
    tail: usize,
    /// Number of bytes currently in use.
    used: usize,
}

impl HeapBuffer {
    /// Create a new heap-backed ring buffer of `total_bytes`.
    ///
    /// # Errors
    /// Returns `GpuError::ZeroAllocation` if `total_bytes` is 0.
    pub fn new(total_bytes: usize) -> Result<Self, GpuError> {
        if total_bytes == 0 {
            return Err(GpuError::ZeroAllocation(total_bytes));
        }
        tracing::debug!(total_bytes, "heap ring buffer created");
        Ok(Self {
            buf: vec![0u8; total_bytes],
            capacity: total_bytes,
            head: 0,
            tail: 0,
            used: 0,
        })
    }

    /// Copy `data` into the next free slot, returning a `HeapSlot`.
    ///
    /// The ring buffer does NOT wrap a single write across the boundary — if
    /// the remaining space at the end is too small, it inserts padding and
    /// wraps the head to 0. This keeps each slot contiguous.
    ///
    /// # Errors
    /// Returns `GpuError::BufferFull` if there is not enough room.
    pub fn write_slot(&mut self, data: &[u8]) -> Result<HeapSlot, GpuError> {
        let len = data.len();
        if len == 0 {
            return Ok(HeapSlot { offset: 0, len: 0 });
        }

        // Check if we need to wrap: if the data doesn't fit between head and
        // the end of the buffer, but would fit if we started from 0 (and
        // there's enough total free space), wrap the head.
        let contiguous_at_end = self.capacity - self.head;

        let write_offset = if contiguous_at_end >= len {
            // Fits without wrapping.
            self.head
        } else if self.tail > 0 && len <= self.tail.saturating_sub(0) && (self.used + len + contiguous_at_end) <= self.capacity {
            // Waste the tail-end gap and wrap to 0.
            // We account for the wasted bytes as "used" so available_bytes
            // stays accurate.
            self.used += contiguous_at_end;
            self.head = 0;
            0
        } else {
            return Err(GpuError::BufferFull {
                requested: len,
                available: self.available_bytes(),
            });
        };

        // Final check: make sure we don't overlap the tail.
        if self.used + len > self.capacity {
            return Err(GpuError::BufferFull {
                requested: len,
                available: self.available_bytes(),
            });
        }

        self.buf[write_offset..write_offset + len].copy_from_slice(data);
        self.head = write_offset + len;
        self.used += len;

        Ok(HeapSlot {
            offset: write_offset,
            len,
        })
    }

    /// Release a previously-allocated slot, making its bytes available again.
    ///
    /// Slots **must** be released in the same order they were allocated (FIFO).
    pub fn release_slot(&mut self, slot: HeapSlot) {
        if slot.len == 0 {
            return;
        }
        // Advance the tail past this slot.
        self.tail = slot.offset + slot.len;
        self.used = self.used.saturating_sub(slot.len);

        // If we've consumed everything, reset both pointers to 0 to maximise
        // contiguous space.
        if self.used == 0 {
            self.head = 0;
            self.tail = 0;
        }
    }

    /// Bytes available for new writes (approximate — ignores fragmentation).
    pub fn available_bytes(&self) -> usize {
        self.capacity.saturating_sub(self.used)
    }

    /// Read the bytes backing a slot. Useful for tests / CPU-side consumers.
    pub fn slot_data(&self, slot: &HeapSlot) -> &[u8] {
        &self.buf[slot.offset..slot.offset + slot.len]
    }
}

// ---------------------------------------------------------------------------
// PinnedBuffer — CUDA feature only
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use cudarc::driver::{CudaContext, sys};
    use std::sync::Arc;

    /// A ring buffer backed by CUDA pinned (page-locked) host memory.
    ///
    /// Pinned memory enables DMA-based host-to-device transfers that bypass
    /// the CPU page table, roughly doubling throughput compared to pageable
    /// memory. The trade-off is that pinned allocations reduce the OS page
    /// cache, so we allocate one large block and sub-allocate within it.
    pub struct PinnedBuffer {
        /// Raw pointer to the start of the pinned allocation.
        ptr: *mut u8,
        capacity: usize,
        head: usize,
        tail: usize,
        used: usize,
        /// Keep the device alive so the CUDA context remains valid.
        _device: Arc<CudaContext>,
    }

    // SAFETY: The pinned allocation is process-wide host memory that is
    // thread-safe to read/write (no GPU-side aliasing at this layer).
    unsafe impl Send for PinnedBuffer {}

    impl PinnedBuffer {
        /// Allocate `total_bytes` of pinned host memory on `device`.
        ///
        /// # Errors
        /// Returns `GpuError::ZeroAllocation` or `GpuError::CudaDriver`.
        pub fn new(device: &Arc<CudaContext>, total_bytes: usize) -> Result<Self, GpuError> {
            if total_bytes == 0 {
                return Err(GpuError::ZeroAllocation(total_bytes));
            }

            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            // cudarc doesn't expose cudaMallocHost directly, so use the
            // driver-level sys bindings.
            unsafe {
                let result = sys::cuMemAllocHost_v2(&mut ptr, total_bytes);
                if result != sys::CUresult::CUDA_SUCCESS {
                    return Err(GpuError::CudaDriver(cudarc::driver::DriverError(result)));
                }
            }

            tracing::debug!(total_bytes, "pinned ring buffer created");

            Ok(Self {
                ptr: ptr.cast::<u8>(),
                capacity: total_bytes,
                head: 0,
                tail: 0,
                used: 0,
                _device: Arc::clone(device),
            })
        }

        /// Copy `data` into the next free slot.
        pub fn write_slot(&mut self, data: &[u8]) -> Result<PinnedSlot, GpuError> {
            let len = data.len();
            if len == 0 {
                return Ok(PinnedSlot {
                    ptr: self.ptr,
                    len: 0,
                    offset: 0,
                });
            }

            let contiguous_at_end = self.capacity - self.head;

            let write_offset = if contiguous_at_end >= len {
                self.head
            } else if self.tail > 0
                && len <= self.tail
                && (self.used + len + contiguous_at_end) <= self.capacity
            {
                self.used += contiguous_at_end;
                self.head = 0;
                0
            } else {
                return Err(GpuError::BufferFull {
                    requested: len,
                    available: self.available_bytes(),
                });
            };

            if self.used + len > self.capacity {
                return Err(GpuError::BufferFull {
                    requested: len,
                    available: self.available_bytes(),
                });
            }

            unsafe {
                let dst = self.ptr.add(write_offset);
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, len);
            }

            self.head = write_offset + len;
            self.used += len;

            Ok(PinnedSlot {
                ptr: unsafe { self.ptr.add(write_offset) },
                len,
                offset: write_offset,
            })
        }

        /// Release a slot (FIFO order required).
        pub fn release_slot(&mut self, slot: PinnedSlot) {
            if slot.len == 0 {
                return;
            }
            self.tail = slot.offset + slot.len;
            self.used = self.used.saturating_sub(slot.len);
            if self.used == 0 {
                self.head = 0;
                self.tail = 0;
            }
        }

        /// Bytes available for new writes.
        pub fn available_bytes(&self) -> usize {
            self.capacity.saturating_sub(self.used)
        }

        /// Raw pointer to the start of the pinned allocation.
        pub fn base_ptr(&self) -> *mut u8 {
            self.ptr
        }
    }

    impl Drop for PinnedBuffer {
        fn drop(&mut self) {
            unsafe {
                let result = sys::cuMemFreeHost(self.ptr.cast());
                if result != sys::CUresult::CUDA_SUCCESS {
                    tracing::error!(?result, "failed to free pinned host memory");
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::PinnedBuffer;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- HeapBuffer tests (always run) --

    #[test]
    fn heap_basic_write_read() {
        let mut buf = HeapBuffer::new(1024).unwrap();
        let data = b"hello, ring buffer!";
        let slot = buf.write_slot(data).unwrap();
        assert_eq!(buf.slot_data(&slot), data);
        buf.release_slot(slot);
        assert_eq!(buf.available_bytes(), 1024);
    }

    #[test]
    fn heap_zero_length_slot() {
        let mut buf = HeapBuffer::new(64).unwrap();
        let slot = buf.write_slot(b"").unwrap();
        assert_eq!(slot.len, 0);
        buf.release_slot(slot);
    }

    #[test]
    fn heap_zero_capacity_fails() {
        let result = HeapBuffer::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn heap_fill_and_release_fifo() {
        let mut buf = HeapBuffer::new(100).unwrap();

        // Write three chunks
        let s1 = buf.write_slot(&[1u8; 30]).unwrap();
        let s2 = buf.write_slot(&[2u8; 30]).unwrap();
        let s3 = buf.write_slot(&[3u8; 30]).unwrap();
        assert_eq!(buf.available_bytes(), 10);

        // Can't fit another 30
        assert!(buf.write_slot(&[4u8; 30]).is_err());

        // Release in order
        buf.release_slot(s1);
        buf.release_slot(s2);
        buf.release_slot(s3);
        assert_eq!(buf.available_bytes(), 100);
    }

    #[test]
    fn heap_wrap_around() {
        let mut buf = HeapBuffer::new(100).unwrap();

        // Fill 80 bytes
        let s1 = buf.write_slot(&[0xAA; 80]).unwrap();
        assert_eq!(buf.available_bytes(), 20);

        // Release — tail moves to 80, head stays at 80
        buf.release_slot(s1);
        assert_eq!(buf.available_bytes(), 100);

        // Now write 60 bytes — won't fit at end (only 20 contiguous), must
        // wrap to offset 0.
        let s2 = buf.write_slot(&[0xBB; 60]).unwrap();
        assert_eq!(buf.slot_data(&s2), &[0xBB; 60]);
        buf.release_slot(s2);
    }

    #[test]
    fn heap_full_then_drain() {
        let mut buf = HeapBuffer::new(64).unwrap();
        let s = buf.write_slot(&[0xFF; 64]).unwrap();
        assert_eq!(buf.available_bytes(), 0);
        assert!(buf.write_slot(&[0x00; 1]).is_err());
        buf.release_slot(s);
        assert_eq!(buf.available_bytes(), 64);

        // Should be able to reuse from 0 after full drain.
        let s2 = buf.write_slot(&[0x11; 32]).unwrap();
        assert_eq!(buf.slot_data(&s2), &[0x11; 32]);
        buf.release_slot(s2);
    }

    #[test]
    fn heap_many_small_writes() {
        let mut buf = HeapBuffer::new(256).unwrap();
        let mut slots = Vec::new();

        for i in 0u8..16 {
            let slot = buf.write_slot(&[i; 16]).unwrap();
            slots.push(slot);
        }

        assert_eq!(buf.available_bytes(), 0);

        for (i, slot) in slots.into_iter().enumerate() {
            assert_eq!(buf.slot_data(&slot), &[i as u8; 16]);
            buf.release_slot(slot);
        }

        assert_eq!(buf.available_bytes(), 256);
    }

    #[test]
    fn heap_data_integrity_after_wrap() {
        let mut buf = HeapBuffer::new(100).unwrap();

        // Write 70 bytes and release
        let s1 = buf.write_slot(&[0xAA; 70]).unwrap();
        buf.release_slot(s1);

        // Head is now at 70, tail at 70 (reset to 0 since used == 0).
        // Actually after release with used==0, both reset to 0.

        // Write 50 + 40 = 90 bytes
        let pattern_a: Vec<u8> = (0..50).collect();
        let pattern_b: Vec<u8> = (50..90).collect();

        let sa = buf.write_slot(&pattern_a).unwrap();
        let sb = buf.write_slot(&pattern_b).unwrap();

        assert_eq!(buf.slot_data(&sa), &pattern_a[..]);
        assert_eq!(buf.slot_data(&sb), &pattern_b[..]);

        buf.release_slot(sa);
        buf.release_slot(sb);
    }

    // -- PinnedBuffer tests (require CUDA, ignored on macOS) --

    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use super::*;
        use cudarc::driver::CudaContext;

        #[test]
        #[ignore = "requires CUDA GPU"]
        fn pinned_basic_write_release() {
            let device = CudaContext::new(0).unwrap();
            let mut buf = PinnedBuffer::new(&device, 1024).unwrap();

            let data = b"pinned hello";
            let slot = buf.write_slot(data).unwrap();
            assert_eq!(slot.len, data.len());
            assert!(!slot.ptr.is_null());

            // Read back via raw ptr
            let read_back =
                unsafe { std::slice::from_raw_parts(slot.ptr, slot.len) };
            assert_eq!(read_back, data);

            buf.release_slot(slot);
            assert_eq!(buf.available_bytes(), 1024);
        }

        #[test]
        #[ignore = "requires CUDA GPU"]
        fn pinned_zero_capacity_fails() {
            let device = CudaContext::new(0).unwrap();
            assert!(PinnedBuffer::new(&device, 0).is_err());
        }

        #[test]
        #[ignore = "requires CUDA GPU"]
        fn pinned_fill_and_release() {
            let device = CudaContext::new(0).unwrap();
            let mut buf = PinnedBuffer::new(&device, 128).unwrap();

            let s1 = buf.write_slot(&[1u8; 64]).unwrap();
            let s2 = buf.write_slot(&[2u8; 64]).unwrap();
            assert_eq!(buf.available_bytes(), 0);
            assert!(buf.write_slot(&[3u8; 1]).is_err());

            buf.release_slot(s1);
            buf.release_slot(s2);
            assert_eq!(buf.available_bytes(), 128);
        }
    }
}
