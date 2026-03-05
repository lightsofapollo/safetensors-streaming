"""Compare sst streaming vs download+load baseline."""
import time

url = "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/model-00001-of-00004.safetensors"

# Baseline: download + safetensors.torch.load_file
import requests, safetensors.torch, tempfile, os

print("Baseline: download + safetensors load_file...")
t0 = time.perf_counter()
resp = requests.get(url, allow_redirects=True)
with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
    f.write(resp.content)
    tmp_path = f.name
result = safetensors.torch.load_file(tmp_path)
os.unlink(tmp_path)
t1 = time.perf_counter()
n = len(result)
elapsed = t1 - t0
print(f"Download+load: {n} tensors in {elapsed:.1f}s")

del result

# Our streaming
import safetensors_streaming as sst
t0 = time.perf_counter()
result = sst.load_file(url)
t1 = time.perf_counter()
n = len(result)
elapsed = t1 - t0
print(f"sst.load_file: {n} tensors in {elapsed:.1f}s")
