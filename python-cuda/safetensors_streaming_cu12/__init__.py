"""CUDA 12 backend for safetensors-streaming.

This package provides the CUDA-enabled native extension. It is auto-detected
by safetensors-streaming when installed.

Install via: pip install safetensors-streaming[cuda]
"""

from safetensors_streaming_cu12.safetensors_streaming import *  # noqa: F401, F403
