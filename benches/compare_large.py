"""Test sst against a larger model to verify consistency.

Models tested:
- Qwen2.5-7B: 4 shards, 15.23 GB (already tested)
- mistralai/Mixtral-8x7B-v0.1: 19 shards, ~87 GB (large, non-gated)
- google/gemma-2-9b: 4 shards, ~18 GB (non-gated)
"""
import time
import gc
import sys

import safetensors_streaming as sst

# Pick model based on arg
models = {
    "gemma-9b": {
        "index": "https://huggingface.co/google/gemma-2-9b/resolve/main/model.safetensors.index.json",
        "single": "https://huggingface.co/google/gemma-2-9b/resolve/main/model-00001-of-00004.safetensors",
    },
    "qwen-7b": {
        "index": "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/model.safetensors.index.json",
        "single": "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/model-00001-of-00004.safetensors",
    },
    "qwen-14b": {
        "index": "https://huggingface.co/Qwen/Qwen2.5-14B/resolve/main/model.safetensors.index.json",
        "single": "https://huggingface.co/Qwen/Qwen2.5-14B/resolve/main/model-00001-of-00005.safetensors",
    },
}

model_name = sys.argv[1] if len(sys.argv) > 1 else "gemma-9b"
model = models[model_name]

print(f"=== Testing {model_name} ===\n")

# Single shard test
print(f"Single shard: load_file...")
t0 = time.perf_counter()
result = sst.load_file(model["single"])
t1 = time.perf_counter()
n = len(result)
total = sum(t.nelement() * t.element_size() for t in result.values())
elapsed = t1 - t0
print(f"  {n} tensors, {total/1e9:.2f} GB in {elapsed:.1f}s ({total/1e6/elapsed:.0f} MB/s)")
del result
gc.collect()

# Full model sharded test
print(f"\nFull model: load_sharded...")
t0 = time.perf_counter()
result = sst.load_sharded(model["index"])
t1 = time.perf_counter()
n = len(result)
total = sum(t.nelement() * t.element_size() for t in result.values())
elapsed = t1 - t0
print(f"  {n} tensors, {total/1e9:.2f} GB in {elapsed:.1f}s ({total/1e6/elapsed:.0f} MB/s)")

# Sanity check: verify tensor values are reasonable
import torch
for name, tensor in list(result.items())[:3]:
    ft = tensor.float() if tensor.dtype == torch.bfloat16 else tensor
    print(f"  {name}: shape={list(tensor.shape)} dtype={tensor.dtype} min={ft.min():.4f} max={ft.max():.4f}")

del result
gc.collect()

# Baseline: sequential download + load
print(f"\nBaseline: sequential download + safetensors.torch.load_file...")
import requests, safetensors.torch, tempfile, os

t0 = time.perf_counter()
resp = requests.get(model["index"], allow_redirects=True)
index = resp.json()
shard_files = sorted(set(index["weight_map"].values()))

all_tensors = {}
for i, shard in enumerate(shard_files):
    shard_url = model["index"].rsplit("/", 1)[0] + "/" + shard
    resp = requests.get(shard_url, allow_redirects=True)
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        f.write(resp.content)
        tmp = f.name
    tensors = safetensors.torch.load_file(tmp)
    all_tensors.update(tensors)
    os.unlink(tmp)
    print(f"  shard {i+1}/{len(shard_files)} loaded")

t1 = time.perf_counter()
elapsed = t1 - t0
n = len(all_tensors)
total = sum(t.nelement() * t.element_size() for t in all_tensors.values())
print(f"  {n} tensors, {total/1e9:.2f} GB in {elapsed:.1f}s ({total/1e6/elapsed:.0f} MB/s)")
