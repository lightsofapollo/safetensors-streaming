"""safetensors_streaming — stream safetensors files from HTTP/S3 directly to GPU memory.

Re-exports the native (Rust/PyO3) module's public API and adds the pure-Python
monkey-patching helpers: ``patch``, ``unpatch``, and ``patched``.
"""

from safetensors_streaming.safetensors_streaming import (  # type: ignore[import-untyped]
    SafeOpen,
    TensorStreamIterator,
    __version__,
    cuda_available,
    load_file,
    safe_open,
    stream_tensors,
    version,
)
from safetensors_streaming.patch import patch, patched, unpatch

__all__ = [
    # Native API
    "SafeOpen",
    "TensorStreamIterator",
    "__version__",
    "cuda_available",
    "load_file",
    "safe_open",
    "stream_tensors",
    "version",
    # Monkey-patching helpers
    "patch",
    "unpatch",
    "patched",
]
