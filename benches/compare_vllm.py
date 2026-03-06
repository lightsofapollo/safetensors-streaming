"""Benchmark: load a model through vLLM with and without sst patching.

vLLM is the most popular inference server — testing that sst works
transparently with its model loading path.
"""
import time
import gc
import sys
import torch

models = {
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen-7b": "Qwen/Qwen2.5-7B",
}

model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen-1.5b"
model_id = models.get(model_name, model_name)
print(f"=== vLLM: {model_id} ===\n")

from huggingface_hub import snapshot_download

# --- Phase 0: Pre-download ---
print("0. Pre-downloading model to cache...")
t0 = time.perf_counter()
snapshot_download(model_id)
t1 = time.perf_counter()
print(f"   Cache ready ({t1 - t0:.1f}s)\n")

# --- Phase 1: Baseline ---
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(max_tokens=10)

print("1. Baseline: standard vLLM load (from cache)...")
t0 = time.perf_counter()
llm = LLM(model=model_id, dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.9, max_model_len=4096)
t1 = time.perf_counter()
baseline_time = t1 - t0
print(f"   Time: {baseline_time:.1f}s")

# Quick inference sanity check
output = llm.generate(["Hello, world!"], sampling_params=sampling_params)
print(f"   Inference check: {output[0].outputs[0].text!r}")

del llm
gc.collect()
torch.cuda.empty_cache()

# --- Phase 2: Patched ---
import safetensors_streaming as sst
sst.patch()

print(f"\n2. Patched: sst safe_open (from cache)...")
t0 = time.perf_counter()
llm = LLM(model=model_id, dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.9, max_model_len=4096)
t1 = time.perf_counter()
patched_time = t1 - t0
print(f"   Time: {patched_time:.1f}s")

sst.unpatch()

# Quick inference sanity check
output = llm.generate(["Hello, world!"], sampling_params=sampling_params)
print(f"   Inference check: {output[0].outputs[0].text!r}")

del llm
gc.collect()
torch.cuda.empty_cache()

# --- Summary ---
print(f"\n=== Results (local cache, apples-to-apples) ===")
print(f"Baseline (safetensors C):  {baseline_time:.1f}s")
print(f"Patched  (sst Rust):      {patched_time:.1f}s")
if patched_time < baseline_time:
    print(f"Speedup: {baseline_time / patched_time:.2f}x faster")
else:
    print(f"Overhead: {patched_time / baseline_time:.2f}x slower")
