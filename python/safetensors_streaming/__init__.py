"""safetensors_streaming — stream safetensors files from HTTP/S3 directly to GPU memory.

Re-exports the native (Rust/PyO3) module's public API and adds the pure-Python
monkey-patching helpers: ``patch``, ``unpatch``, and ``patched``.

Auto-detects CUDA support: if ``safetensors-streaming-cu12`` is installed,
the CUDA-enabled backend is used automatically. Otherwise falls back to CPU.
"""

# Auto-detect: prefer CUDA backend if installed, fall back to CPU-only
try:
    from safetensors_streaming_cu12.safetensors_streaming import (  # type: ignore[import-untyped]
        SafeOpen,
        ShardedTensorStreamIterator,
        TensorStreamIterator,
        __version__,
        cuda_available,
        load_file,
        load_sharded,
        safe_open,
        stream_sharded,
        stream_tensors,
        version,
    )
except ImportError:
    from safetensors_streaming.safetensors_streaming import (  # type: ignore[import-untyped]
        SafeOpen,
        ShardedTensorStreamIterator,
        TensorStreamIterator,
        __version__,
        cuda_available,
        load_file,
        load_sharded,
        safe_open,
        stream_sharded,
        stream_tensors,
        version,
    )

from safetensors_streaming.patch import patch, patched, unpatch
from safetensors_streaming.progress import tqdm_load_file, tqdm_stream_tensors

__all__ = [
    # Native API
    "SafeOpen",
    "ShardedTensorStreamIterator",
    "TensorStreamIterator",
    "__version__",
    "cuda_available",
    "load_file",
    "load_sharded",
    "safe_open",
    "stream_sharded",
    "stream_tensors",
    "version",
    # Monkey-patching helpers
    "patch",
    "unpatch",
    "patched",
    # Progress-reporting wrappers
    "tqdm_load_file",
    "tqdm_stream_tensors",
]
