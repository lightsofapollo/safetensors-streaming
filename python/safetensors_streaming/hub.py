"""Monkey-patch huggingface_hub to stream safetensors files instead of downloading them.

When patched, calls to ``hf_hub_download`` for ``.safetensors`` files return a
HuggingFace resolve URL instead of downloading the file to disk.  Combined with
``safetensors_streaming.load_file`` or ``safe_open`` (which accept URLs), this
enables streaming model weights directly from the Hub without a full download.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Callable, Iterator

# Sentinel indicating that the hub has not been patched yet.
_original_hf_hub_download: Callable[..., str] | None = None


def _resolve_url(
    repo_id: str,
    filename: str,
    *,
    revision: str = "main",
    subfolder: str | None = None,
) -> str:
    """Build a HuggingFace resolve URL for the given file.

    Args:
        repo_id: Repository identifier, e.g. ``"meta-llama/Llama-2-7b"``.
        filename: File name within the repository.
        revision: Git revision (branch, tag, or commit hash). Defaults to ``"main"``.
        subfolder: Optional subfolder within the repository.

    Returns:
        The full resolve URL string.
    """
    if subfolder:
        path = f"{subfolder}/{filename}"
    else:
        path = filename
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{path}"


def _make_patched_download(
    original: Callable[..., str],
) -> Callable[..., str]:
    """Create a patched version of ``hf_hub_download`` that intercepts safetensors requests.

    Args:
        original: The original ``hf_hub_download`` function.

    Returns:
        A wrapper function with the same signature.
    """

    def patched_hf_hub_download(
        repo_id: str,
        filename: str,
        *args: object,
        subfolder: str | None = None,
        revision: str | None = None,
        repo_type: str | None = None,
        **kwargs: object,
    ) -> str:
        """Patched ``hf_hub_download`` that returns a resolve URL for safetensors files.

        For ``.safetensors`` files in model repos, this returns the HuggingFace
        resolve URL directly instead of downloading. For all other files, it
        delegates to the original ``hf_hub_download``.
        """
        # Only intercept model repos (or default None which means model)
        is_model_repo = repo_type is None or repo_type == "model"

        if is_model_repo and filename.endswith(".safetensors"):
            effective_revision = revision if revision is not None else "main"
            return _resolve_url(
                repo_id,
                filename,
                revision=effective_revision,
                subfolder=subfolder,
            )

        # Fall through to original for non-safetensors files or non-model repos
        return original(
            repo_id,
            filename,
            *args,
            subfolder=subfolder,
            revision=revision,
            repo_type=repo_type,
            **kwargs,
        )

    return patched_hf_hub_download


def _get_huggingface_hub() -> types.ModuleType:
    """Import and return the ``huggingface_hub`` module.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        import huggingface_hub  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for hub patching. "
            "Install it with: pip install huggingface_hub"
        ) from None
    return huggingface_hub


def patch_hub() -> None:
    """Patch ``huggingface_hub.hf_hub_download`` to stream safetensors files.

    After calling this, any code that uses ``hf_hub_download`` to fetch a
    ``.safetensors`` file from a model repo will receive a resolve URL instead
    of a local file path. This URL can be passed directly to
    ``safetensors_streaming.load_file`` or ``safe_open`` for streaming.

    Call :func:`unpatch_hub` to restore the original behavior.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        RuntimeError: If the hub is already patched.
    """
    global _original_hf_hub_download

    if _original_hf_hub_download is not None:
        raise RuntimeError(
            "huggingface_hub is already patched. Call unpatch_hub() first."
        )

    hub = _get_huggingface_hub()
    _original_hf_hub_download = hub.hf_hub_download  # type: ignore[assignment]
    hub.hf_hub_download = _make_patched_download(_original_hf_hub_download)  # type: ignore[assignment]


def unpatch_hub() -> None:
    """Restore the original ``huggingface_hub.hf_hub_download``.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        RuntimeError: If the hub is not currently patched.
    """
    global _original_hf_hub_download

    if _original_hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub is not patched. Call patch_hub() first."
        )

    hub = _get_huggingface_hub()
    hub.hf_hub_download = _original_hf_hub_download  # type: ignore[assignment]
    _original_hf_hub_download = None


@contextmanager
def patched_hub() -> Iterator[None]:
    """Context manager that temporarily patches ``huggingface_hub.hf_hub_download``.

    Usage::

        with patched_hub():
            # hf_hub_download returns URLs for .safetensors files
            path = huggingface_hub.hf_hub_download("meta-llama/Llama-2-7b", "model.safetensors")
            # path is now a URL that can be streamed
            tensors = safetensors_streaming.load_file(path)

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    patch_hub()
    try:
        yield
    finally:
        unpatch_hub()
