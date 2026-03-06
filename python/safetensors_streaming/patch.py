"""Monkey-patching helpers for transparently replacing ``safetensors`` with streaming versions.

Usage::

    import safetensors_streaming

    # Permanent patch (until unpatch() or process exit)
    safetensors_streaming.patch()

    # Scoped patch via context manager
    with safetensors_streaming.patched():
        model = AutoModel.from_pretrained("gpt2")
"""

from __future__ import annotations

import contextlib
import types
from collections.abc import Generator
from typing import Callable

# ---------------------------------------------------------------------------
# Internal state — stores original functions so we can restore them
# ---------------------------------------------------------------------------

_originals: dict[str, Callable[..., object]] = {}
_patched: bool = False


def _ensure_safetensors_importable() -> None:
    """Raise a clear ``ImportError`` if the ``safetensors`` package is missing."""
    try:
        import safetensors  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'safetensors' package must be installed before patching. "
            "Install it with:  pip install safetensors"
        ) from exc


def _get_module(dotted_name: str) -> types.ModuleType:
    """Import and return *dotted_name* (e.g. ``safetensors.torch``)."""
    import importlib

    return importlib.import_module(dotted_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def patch() -> None:
    """Replace ``safetensors.safe_open`` and ``safetensors.torch.load_file`` with streaming versions.

    Idempotent — calling this more than once is safe (subsequent calls are no-ops).
    """
    global _patched  # noqa: PLW0603

    if _patched:
        return

    _ensure_safetensors_importable()

    # Import our native implementations (already validated by the __init__ re-export).
    from safetensors_streaming.safetensors_streaming import (  # type: ignore[import-untyped]
        load_file as _streaming_load_file,
        safe_open as _streaming_safe_open,
    )

    # --- safetensors.safe_open ---
    sf_mod = _get_module("safetensors")
    _originals["safetensors.safe_open"] = sf_mod.safe_open  # type: ignore[attr-defined]
    sf_mod.safe_open = _streaming_safe_open  # type: ignore[attr-defined]

    # --- safetensors.torch.load_file ---
    # For URLs, use our streaming load_file. For local files, delegate to the
    # native implementation — it uses mmap + zero-copy which is unbeatable.
    try:
        sf_torch_mod = _get_module("safetensors.torch")
        _originals["safetensors.torch.load_file"] = sf_torch_mod.load_file  # type: ignore[attr-defined]
        _native_load_file = sf_torch_mod.load_file  # type: ignore[attr-defined]

        def _hybrid_load_file(path_or_url: str, *, device: str = "cpu") -> dict:  # type: ignore[type-arg]
            if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
                return _streaming_load_file(path_or_url, device=device)  # type: ignore[no-any-return]
            return _native_load_file(path_or_url, device=device)  # type: ignore[no-any-return]

        sf_torch_mod.load_file = _hybrid_load_file  # type: ignore[attr-defined]
    except ImportError:
        # safetensors installed without torch extras — skip torch patching
        pass

    _patched = True


def unpatch() -> None:
    """Restore the original ``safetensors`` functions.

    Safe to call even if :func:`patch` was never called (no-op in that case).
    """
    global _patched  # noqa: PLW0603

    if not _patched:
        return

    # --- safetensors.safe_open ---
    original_safe_open = _originals.pop("safetensors.safe_open", None)
    if original_safe_open is not None:
        sf_mod = _get_module("safetensors")
        sf_mod.safe_open = original_safe_open  # type: ignore[attr-defined]

    # --- safetensors.torch.load_file ---
    original_load_file = _originals.pop("safetensors.torch.load_file", None)
    if original_load_file is not None:
        sf_torch_mod = _get_module("safetensors.torch")
        sf_torch_mod.load_file = original_load_file  # type: ignore[attr-defined]

    _patched = False


@contextlib.contextmanager
def patched() -> Generator[None, None, None]:
    """Context manager that patches on entry and restores on exit.

    Example::

        with safetensors_streaming.patched():
            model = AutoModel.from_pretrained("gpt2")
    """
    patch()
    try:
        yield
    finally:
        unpatch()
