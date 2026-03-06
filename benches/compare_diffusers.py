"""Benchmark: load a diffusion model through diffusers with and without sst patching.

Tests SDXL and Stable Diffusion pipelines to verify sst works as a drop-in
replacement across the diffusers ecosystem.
"""
import time
import gc
import sys
import torch

models = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd-turbo": "stabilityai/sd-turbo",
    "sd-1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
}

model_name = sys.argv[1] if len(sys.argv) > 1 else "sd-turbo"
model_id = models.get(model_name, model_name)
print(f"=== Diffusers: {model_id} ===\n")

from huggingface_hub import snapshot_download

# --- Phase 0: Pre-download ---
print("0. Pre-downloading model to cache...")
t0 = time.perf_counter()
snapshot_download(model_id)
t1 = time.perf_counter()
print(f"   Cache ready ({t1 - t0:.1f}s)\n")

# --- Phase 1: Baseline load ---
from diffusers import DiffusionPipeline

print("1. Baseline: standard diffusers load (from cache)...")
t0 = time.perf_counter()
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)
t1 = time.perf_counter()
baseline_time = t1 - t0

# Count params across all components
total_params = 0
total_bytes = 0
components = []
for name, component in pipe.components.items():
    if hasattr(component, 'parameters'):
        p = sum(p.numel() for p in component.parameters())
        b = sum(p.numel() * p.element_size() for p in component.parameters())
        total_params += p
        total_bytes += b
        components.append(f"{name}({p/1e6:.0f}M)")

print(f"   Components: {', '.join(components)}")
print(f"   Total: {total_params/1e9:.2f}B params, {total_bytes/1e9:.2f} GB")
print(f"   Time: {baseline_time:.1f}s")

# Save baseline state for verification (just UNet/transformer — the big model)
big_component = None
for name in ['unet', 'transformer']:
    comp = getattr(pipe, name, None)
    if comp is not None:
        big_component = name
        break

if big_component:
    baseline_state = {k: v.clone() for k, v in getattr(pipe, big_component).state_dict().items()}
    print(f"   Saved {len(baseline_state)} {big_component} weights for verification")

del pipe
gc.collect()

# --- Phase 2: Patched load ---
import safetensors_streaming as sst
sst.patch()

print(f"\n2. Patched: sst safe_open (from cache)...")
t0 = time.perf_counter()
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)
t1 = time.perf_counter()
patched_time = t1 - t0
print(f"   Time: {patched_time:.1f}s")

sst.unpatch()

# --- Phase 3: Verify ---
if big_component:
    patched_state = getattr(pipe, big_component).state_dict()
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
        print(f"   All {len(baseline_state)} {big_component} tensors match!")
    else:
        print(f"   WARNING: {mismatches} mismatches out of {len(baseline_state)}")

del pipe
if big_component:
    del baseline_state, patched_state
gc.collect()

# --- Summary ---
print(f"\n=== Results (local cache, apples-to-apples) ===")
print(f"Baseline (safetensors C):  {baseline_time:.1f}s")
print(f"Patched  (sst Rust):      {patched_time:.1f}s")
if patched_time < baseline_time:
    print(f"Speedup: {baseline_time / patched_time:.2f}x faster")
else:
    print(f"Overhead: {patched_time / baseline_time:.2f}x slower")
