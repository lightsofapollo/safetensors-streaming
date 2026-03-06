"""Real-world benchmark: load a model through transformers with and without sst patching.

Tests that our streaming library works as a transparent drop-in replacement
for safetensors.torch.load_file when used with HuggingFace transformers.

Phase 1: Pre-download model to cache (excluded from timing)
Phase 2: Baseline load from cache (standard safetensors)
Phase 3: Patched load from cache (our safe_open)
Phase 4: Verify all tensors match
"""
import time
import gc
import sys
import torch

models = {
    "qwen-7b": "Qwen/Qwen2.5-7B",
    "gemma-9b": "google/gemma-2-9b",
    "qwen-14b": "Qwen/Qwen2.5-14B",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
}

model_id = models.get(sys.argv[1], sys.argv[1]) if len(sys.argv) > 1 else models["qwen-7b"]
print(f"=== Model: {model_id} ===\n")

from transformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import snapshot_download

# Get model config to show size
config = AutoConfig.from_pretrained(model_id)
print(f"Config: {config.num_hidden_layers} layers, {config.hidden_size} hidden, {config.vocab_size} vocab\n")

# --- Phase 1: Pre-download to cache (not timed) ---
print("0. Pre-downloading model to cache...")
t0 = time.perf_counter()
snapshot_download(model_id)
t1 = time.perf_counter()
print(f"   Cache ready ({t1 - t0:.1f}s)\n")

# --- Phase 2: Baseline load from cache ---
print("1. Baseline: standard safetensors (from cache)...")
t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
t1 = time.perf_counter()
baseline_time = t1 - t0

param_count = sum(p.numel() for p in model.parameters())
param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"   {param_count/1e9:.2f}B params, {param_bytes/1e9:.2f} GB")
print(f"   Time: {baseline_time:.1f}s")

# Quick sanity: check a weight
first_param_name = next(iter(model.state_dict().keys()))
first_param = model.state_dict()[first_param_name]
print(f"   Sample: {first_param_name} shape={list(first_param.shape)} dtype={first_param.dtype}")
baseline_state = {k: v.clone() for k, v in model.state_dict().items()}

del model
gc.collect()

# --- Phase 3: Patched load from cache ---
print("\n2. Patched: sst safe_open (from cache)...")
import safetensors_streaming as sst

sst.patch()

t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
t1 = time.perf_counter()
patched_time = t1 - t0

print(f"   Time: {patched_time:.1f}s")

sst.unpatch()

# --- Phase 4: Verify correctness ---
patched_state = model.state_dict()
mismatches = 0
for key in baseline_state:
    if key not in patched_state:
        print(f"   MISSING: {key}")
        mismatches += 1
        continue
    if not torch.equal(baseline_state[key], patched_state[key]):
        print(f"   MISMATCH: {key}")
        mismatches += 1

if mismatches == 0:
    print(f"   All {len(baseline_state)} tensors match!")
else:
    print(f"   WARNING: {mismatches} mismatches out of {len(baseline_state)}")

del model, baseline_state, patched_state
gc.collect()

# --- Summary ---
print(f"\n=== Results (local cache, apples-to-apples) ===")
print(f"Baseline (safetensors C):  {baseline_time:.1f}s")
print(f"Patched  (sst Rust):      {patched_time:.1f}s")
if patched_time < baseline_time:
    print(f"Speedup: {baseline_time / patched_time:.2f}x faster")
else:
    print(f"Overhead: {patched_time / baseline_time:.2f}x slower")
