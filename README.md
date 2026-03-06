# safetensors-streaming

Drop-in Rust replacement for `safetensors.safe_open()` and `safetensors.torch.load_file()` that loads model weights faster by bypassing Python overhead.

> **Experimental** — this project is under active development. APIs may change. Use at your own risk.

## What it does

The standard `safetensors` Python library reads weights through a C extension with Python-level iteration. `safetensors-streaming` reimplements the hot path in Rust via PyO3, returning tensors directly to PyTorch with less overhead.

It works as a monkey-patch — two lines of code and every library that uses `safetensors` (transformers, diffusers, vLLM, etc.) automatically uses the faster path.

## Usage

```python
import safetensors_streaming

# Patch safetensors globally
safetensors_streaming.patch()

# Everything downstream uses the faster implementation
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
```

Or use a context manager for scoped patching:

```python
import safetensors_streaming

with safetensors_streaming.patched():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
```

## Benchmarks

All benchmarks run on an RTX 4090, loading from local cache (no network). Each test loads the model twice — once with the stock `safetensors` library, once with `safetensors-streaming` patched in.

| Library | Model | Params | Baseline | Patched | Speedup |
|---------|-------|--------|----------|---------|---------|
| transformers | Phi-3-mini-4k | 3.8B | 2.0s | 0.2s | **8.4x** |
| transformers | Qwen2.5-7B | 7.6B | 2.2s | 0.3s | **7.1x** |
| transformers | Qwen2.5-14B | 14.8B | 2.8s | 0.7s | **3.9x** |
| diffusers | SD-Turbo | 1.3B | 4.5s | 2.5s | **1.77x** |
| sentence-transformers | bge-large-en | 335M | 1.33s | 1.03s | **1.29x** |
| vLLM | Qwen2.5-7B | 7.6B | 13.2s | 11.0s | **1.21x** |

The biggest gains are in `transformers` where weight loading dominates total time. In vLLM, engine initialization (process spawning, KV cache allocation, kernel warmup) accounts for ~75% of startup, so the weight loading speedup has less overall impact.

## How it works

The stock `safetensors` Python library uses `safe_open()` which returns a handle, then Python code calls `get_tensor()` in a loop for each tensor. Each call crosses the Python/C boundary and allocates individually.

`safetensors-streaming` parses the safetensors header in Rust, then reads all tensor data in bulk with minimal Python interaction. For local files it uses memory-mapped I/O. The Rust implementation also supports HTTP Range requests and S3 for streaming weights directly from remote storage.

## Install

Requires Python 3.10+, PyTorch, and a Rust toolchain (1.80+).

```bash
git clone https://github.com/lightsofapollo/safetensors-streaming.git
cd safetensors-streaming
pip install maturin
maturin develop --release
```

This compiles the Rust extension and installs it into your current Python environment.

## Project status

This is experimental software. Known limitations:

- Only tested with `float16`, `bfloat16`, and `float32` dtypes
- Streaming from URLs (HTTP/S3) works but is not yet optimized for production use
- The monkey-patching approach covers `safe_open()` and `load_file()` — other safetensors APIs are not patched

## License

MIT

## Credits

Built by [lightsofapollo](https://github.com/lightsofapollo).
