"""Benchmark: load a model through sentence-transformers with and without sst patching.

sentence-transformers is widely used for embeddings and RAG.
"""
import time
import gc
import sys
import torch

models = {
    "bge-large": "BAAI/bge-large-en-v1.5",
    "e5-large": "intfloat/e5-large-v2",
    "gte-large": "thenlper/gte-large",
}

model_name = sys.argv[1] if len(sys.argv) > 1 else "bge-large"
model_id = models.get(model_name, model_name)
print(f"=== Sentence-Transformers: {model_id} ===\n")

from huggingface_hub import snapshot_download

# --- Phase 0: Pre-download ---
print("0. Pre-downloading model to cache...")
t0 = time.perf_counter()
snapshot_download(model_id)
t1 = time.perf_counter()
print(f"   Cache ready ({t1 - t0:.1f}s)\n")

# --- Phase 1: Baseline ---
from sentence_transformers import SentenceTransformer

print("1. Baseline: standard sentence-transformers load (from cache)...")
t0 = time.perf_counter()
model = SentenceTransformer(model_id, device="cpu")
t1 = time.perf_counter()
baseline_time = t1 - t0

# Count params
total_params = sum(p.numel() for p in model.parameters())
total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"   {total_params/1e6:.0f}M params, {total_bytes/1e6:.0f} MB")
print(f"   Time: {baseline_time:.3f}s")

# Sanity check embedding
emb1 = model.encode(["Hello world"], convert_to_tensor=True)
print(f"   Embedding shape: {list(emb1.shape)}, norm: {emb1.norm():.4f}")

del model
gc.collect()

# --- Phase 2: Patched ---
import safetensors_streaming as sst
sst.patch()

print(f"\n2. Patched: sst safe_open (from cache)...")
t0 = time.perf_counter()
model = SentenceTransformer(model_id, device="cpu")
t1 = time.perf_counter()
patched_time = t1 - t0
print(f"   Time: {patched_time:.3f}s")

# Verify embedding matches
emb2 = model.encode(["Hello world"], convert_to_tensor=True)
print(f"   Embedding shape: {list(emb2.shape)}, norm: {emb2.norm():.4f}")

sst.unpatch()

# Compare embeddings
cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
print(f"   Cosine similarity: {cos_sim:.6f}")
if cos_sim > 0.9999:
    print(f"   Embeddings match!")
else:
    print(f"   WARNING: embeddings differ (cos_sim={cos_sim:.6f})")

del model
gc.collect()

# --- Summary ---
print(f"\n=== Results (local cache, apples-to-apples) ===")
print(f"Baseline (safetensors C):  {baseline_time:.3f}s")
print(f"Patched  (sst Rust):      {patched_time:.3f}s")
if patched_time < baseline_time:
    print(f"Speedup: {baseline_time / patched_time:.2f}x faster")
else:
    print(f"Overhead: {patched_time / baseline_time:.2f}x slower")
