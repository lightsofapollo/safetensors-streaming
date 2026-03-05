"""Progress-reporting wrappers for safetensors-streaming.

Provides ``tqdm_load_file`` and ``tqdm_stream_tensors`` which wrap the core
API with a progress bar (tqdm if installed, otherwise a simple print-based
fallback).

Usage::

    from safetensors_streaming.progress import tqdm_load_file, tqdm_stream_tensors

    # Load all tensors with progress
    tensors = tqdm_load_file("https://hf.co/.../model.safetensors")

    # Stream tensors with per-tensor progress
    for name, tensor in tqdm_stream_tensors("https://hf.co/.../model.safetensors"):
        model.load_state_dict({name: tensor}, strict=False)
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Progress bar abstraction
# ---------------------------------------------------------------------------


class ProgressBar(Protocol):
    """Minimal interface that both tqdm and our fallback implement."""

    def update(self, n: int = 1) -> None: ...
    def set_postfix_str(self, s: str) -> None: ...
    def close(self) -> None: ...


class _TqdmWrapper:
    """Thin wrapper around a real tqdm bar."""

    def __init__(self, total: int | None, desc: str, unit: str) -> None:
        import tqdm as _tqdm_mod

        self._bar: _tqdm_mod.tqdm[None] = _tqdm_mod.tqdm(
            total=total, desc=desc, unit=unit, dynamic_ncols=True
        )

    def update(self, n: int = 1) -> None:
        self._bar.update(n)

    def set_postfix_str(self, s: str) -> None:
        self._bar.set_postfix_str(s)

    def close(self) -> None:
        self._bar.close()


class _PrintProgress:
    """Fallback progress reporter using plain prints to stderr."""

    def __init__(self, total: int | None, desc: str, unit: str) -> None:
        self._desc = desc
        self._unit = unit
        self._total = total
        self._current = 0
        self._postfix = ""
        self._last_print = 0.0

    def update(self, n: int = 1) -> None:
        self._current += n
        now = time.monotonic()
        # Throttle output to at most once per 0.25s
        if now - self._last_print < 0.25:
            return
        self._last_print = now
        self._print_line()

    def set_postfix_str(self, s: str) -> None:
        self._postfix = s

    def close(self) -> None:
        self._print_line()
        print(file=sys.stderr)  # newline

    def _print_line(self) -> None:
        if self._total is not None:
            pct = self._current / self._total * 100 if self._total > 0 else 100.0
            line = f"\r{self._desc}: {self._current}/{self._total} {self._unit} ({pct:.0f}%)"
        else:
            line = f"\r{self._desc}: {self._current} {self._unit}"
        if self._postfix:
            line += f" | {self._postfix}"
        print(line, end="", file=sys.stderr, flush=True)


def _make_bar(total: int | None, desc: str, unit: str) -> ProgressBar:
    """Create a tqdm bar if available, otherwise a print-based fallback."""
    try:
        import tqdm as _tqdm_mod  # noqa: F401

        return _TqdmWrapper(total=total, desc=desc, unit=unit)
    except ImportError:
        return _PrintProgress(total=total, desc=desc, unit=unit)


def _format_bytes(n: int) -> str:
    """Human-readable byte size."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tqdm_load_file(
    url_or_path: str,
    *,
    device: str = "cpu",
    progress_bar: bool = True,
) -> dict[str, torch.Tensor]:
    """Load all tensors from a safetensors file with progress reporting.

    Since ``load_file`` fetches everything before returning, this function
    shows a spinner/status during the download phase and then reports summary
    statistics.

    Parameters
    ----------
    url_or_path:
        Local file path or HTTP/HTTPS URL to a ``.safetensors`` file.
    device:
        Target device (``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.).
    progress_bar:
        If ``False``, behaves identically to ``load_file`` with no output.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping of tensor names to tensors on the requested device.
    """
    from safetensors_streaming.safetensors_streaming import (  # type: ignore[import-untyped]
        load_file,
    )

    if not progress_bar:
        return load_file(url_or_path, device=device)  # type: ignore[no-any-return]

    bar = _make_bar(total=None, desc="Loading", unit="tensors")
    bar.set_postfix_str(f"downloading {url_or_path.rsplit('/', maxsplit=1)[-1]}")

    t0 = time.monotonic()
    result: dict[str, torch.Tensor] = load_file(url_or_path, device=device)  # type: ignore[assignment]
    elapsed = time.monotonic() - t0

    total_bytes = 0
    for tensor in result.values():
        total_bytes += tensor.nelement() * tensor.element_size()

    bar.update(len(result))
    bar.set_postfix_str(
        f"{len(result)} tensors, {_format_bytes(total_bytes)}, {elapsed:.2f}s"
    )
    bar.close()

    return result


def tqdm_stream_tensors(
    url_or_path: str,
    *,
    device: str = "cpu",
    progress_bar: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream tensors from a safetensors file with per-tensor progress.

    Wraps ``stream_tensors`` and updates a progress bar after each tensor
    is received.

    Parameters
    ----------
    url_or_path:
        Local file path or HTTP/HTTPS URL to a ``.safetensors`` file.
    device:
        Target device (``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.).
    progress_bar:
        If ``False``, behaves identically to ``stream_tensors`` with no output.

    Yields
    ------
    tuple[str, torch.Tensor]
        ``(name, tensor)`` pairs in file order.
    """
    from safetensors_streaming.safetensors_streaming import (  # type: ignore[import-untyped]
        stream_tensors,
    )

    if not progress_bar:
        yield from stream_tensors(url_or_path, device=device)
        return

    bar = _make_bar(total=None, desc="Streaming", unit="tensors")
    total_bytes = 0
    count = 0

    for name, tensor in stream_tensors(url_or_path, device=device):
        tensor_bytes = tensor.nelement() * tensor.element_size()
        total_bytes += tensor_bytes
        count += 1

        bar.update(1)
        bar.set_postfix_str(f"{name} | {_format_bytes(total_bytes)} total")
        yield name, tensor

    bar.set_postfix_str(f"done: {count} tensors, {_format_bytes(total_bytes)}")
    bar.close()
