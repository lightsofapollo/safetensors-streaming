"""Compare sst sharded streaming vs download+load baseline for full Qwen2.5-7B."""
import time
import gc

index_url = "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/model.safetensors.index.json"

# Our streaming (all shards concurrent)
import safetensors_streaming as sst

print("sst.load_sharded (all shards concurrent)...")
t0 = time.perf_counter()
result = sst.load_sharded(index_url)
t1 = time.perf_counter()
n = len(result)
total_bytes = sum(t.nelement() * t.element_size() for t in result.values())
elapsed = t1 - t0
print(f"sst.load_sharded: {n} tensors, {total_bytes/1e9:.2f} GB in {elapsed:.1f}s ({total_bytes/1e6/elapsed:.0f} MB/s)")

del result
gc.collect()

# Baseline: download shards sequentially + safetensors.torch.load_file
import requests, safetensors.torch, tempfile, os, json

print("\nBaseline: sequential download + safetensors.torch.load_file...")
t0 = time.perf_counter()

# Fetch index
resp = requests.get(index_url, allow_redirects=True)
index = resp.json()
shard_files = sorted(set(index["weight_map"].values()))

all_tensors = {}
for shard in shard_files:
    shard_url = index_url.rsplit("/", 1)[0] + "/" + shard
    resp = requests.get(shard_url, allow_redirects=True)
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        f.write(resp.content)
        tmp = f.name
    tensors = safetensors.torch.load_file(tmp)
    all_tensors.update(tensors)
    os.unlink(tmp)

t1 = time.perf_counter()
elapsed = t1 - t0
n = len(all_tensors)
total_bytes = sum(t.nelement() * t.element_size() for t in all_tensors.values())
print(f"Baseline: {n} tensors, {total_bytes/1e9:.2f} GB in {elapsed:.1f}s ({total_bytes/1e6/elapsed:.0f} MB/s)")
