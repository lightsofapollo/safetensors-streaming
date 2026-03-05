#!/usr/bin/env python3
"""
safetensors-streaming benchmark suite.

Compares safetensors_streaming against the standard safetensors + huggingface_hub
workflow for loading model weights from local disk and remote URLs.

Usage:
    python benches/benchmark.py                    # run all benchmarks with small model
    python benches/benchmark.py --model medium     # use gpt2 (~548MB)
    python benches/benchmark.py --model all        # run both small and medium
    python benches/benchmark.py --skip-remote      # skip remote/URL benchmarks
    python benches/benchmark.py --skip-gpu         # skip GPU transfer benchmarks
    python benches/benchmark.py --runs 5           # 5 runs per benchmark (default 3)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODELS = {
    "small": {
        "name": "bert_uncased_L-2_H-128_A-2",
        "url": "https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/main/model.safetensors",
        "hf_repo": "google/bert_uncased_L-2_H-128_A-2",
        "hf_filename": "model.safetensors",
    },
    "medium": {
        "name": "gpt2",
        "url": "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors",
        "hf_repo": "openai-community/gpt2",
        "hf_filename": "model.safetensors",
    },
}

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------


def check_safetensors_streaming() -> bool:
    try:
        import safetensors_streaming  # noqa: F401

        return True
    except ImportError:
        return False


def check_safetensors() -> bool:
    try:
        import safetensors.torch  # noqa: F401

        return True
    except ImportError:
        return False


def check_huggingface_hub() -> bool:
    try:
        import huggingface_hub  # noqa: F401

        return True
    except ImportError:
        return False


def check_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def check_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def preflight() -> list[str]:
    """Return list of missing critical dependencies."""
    missing: list[str] = []
    if not check_torch():
        missing.append("torch")
    if not check_safetensors_streaming():
        missing.append("safetensors_streaming")
    if not check_safetensors():
        missing.append("safetensors")
    if not check_huggingface_hub():
        missing.append("huggingface_hub")
    return missing


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def timed(fn: Callable[[], object]) -> float:
    """Run fn and return elapsed wall-clock seconds."""
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def median_of(fn: Callable[[], float], runs: int) -> float:
    """Run fn `runs` times and return the median result."""
    return statistics.median(fn() for _ in range(runs))


def format_time(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


def format_throughput(size_bytes: int, seconds: float) -> str:
    if seconds <= 0:
        return "N/A"
    gb_per_s = (size_bytes / 1e9) / seconds
    return f"{gb_per_s:.2f} GB/s"


def format_ratio(baseline: float, contender: float) -> str:
    if contender <= 0 or baseline <= 0:
        return "N/A"
    ratio = baseline / contender
    if ratio >= 1.0:
        return f"{ratio:.2f}x faster"
    return f"{1 / ratio:.2f}x slower"


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    benchmark: str
    model: str
    metrics: dict[str, float | str] = field(default_factory=dict)


@dataclass
class BenchmarkRun:
    timestamp: str
    model: str
    file_size_bytes: int
    tensor_count: int
    cuda_available: bool
    results: list[BenchmarkResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def download_model(url: str, dest: str) -> None:
    """Download a file from a URL to a local path using urllib."""
    import urllib.request

    urllib.request.urlretrieve(url, dest)


def download_via_hf_hub(repo_id: str, filename: str, cache_dir: str) -> str:
    """Download using huggingface_hub and return local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_local_load(
    local_path: str, file_size: int, runs: int
) -> list[BenchmarkResult]:
    """Compare safetensors.torch.load_file() vs safetensors_streaming.load_file() on a local file."""
    import safetensors.torch
    import safetensors_streaming

    results: list[BenchmarkResult] = []

    def time_safetensors() -> float:
        return timed(lambda: safetensors.torch.load_file(local_path))

    def time_streaming() -> float:
        return timed(lambda: safetensors_streaming.load_file(local_path))

    t_st = median_of(time_safetensors, runs)
    t_sst = median_of(time_streaming, runs)

    ratio = t_st / t_sst if t_sst > 0 else 0.0

    print(f"\nLocal load (from disk):")
    print(
        f"  safetensors.torch.load_file():     {format_time(t_st):>10}  ({format_throughput(file_size, t_st)})"
    )
    print(
        f"  safetensors_streaming.load_file(): {format_time(t_sst):>10}  ({ratio:.2f}x)  ({format_throughput(file_size, t_sst)})"
    )

    results.append(
        BenchmarkResult(
            benchmark="local_load_safetensors",
            model="",
            metrics={
                "seconds": round(t_st, 6),
                "throughput_gbps": round((file_size / 1e9) / t_st, 4) if t_st > 0 else 0,
            },
        )
    )
    results.append(
        BenchmarkResult(
            benchmark="local_load_streaming",
            model="",
            metrics={
                "seconds": round(t_sst, 6),
                "throughput_gbps": round((file_size / 1e9) / t_sst, 4) if t_sst > 0 else 0,
                "ratio_vs_safetensors": round(ratio, 4),
            },
        )
    )
    return results


def bench_remote_load(
    model_info: dict[str, str], runs: int
) -> list[BenchmarkResult]:
    """Compare download+load_file vs streaming load_file from URL."""
    import safetensors.torch
    import safetensors_streaming

    url = model_info["url"]
    repo_id = model_info["hf_repo"]
    filename = model_info["hf_filename"]
    results: list[BenchmarkResult] = []

    def time_download_then_load() -> float:
        with tempfile.TemporaryDirectory() as tmp:
            start = time.perf_counter()
            local_path = download_via_hf_hub(repo_id, filename, cache_dir=tmp)
            safetensors.torch.load_file(local_path)
            return time.perf_counter() - start

    def time_streaming_load() -> float:
        return timed(lambda: safetensors_streaming.load_file(url))

    t_dl = median_of(time_download_then_load, runs)
    t_sst = median_of(time_streaming_load, runs)

    print(f"\nRemote load (from HF):")
    print(f"  download + load_file():            {format_time(t_dl):>10}")
    print(
        f"  streaming load_file(url):           {format_time(t_sst):>10}  ({format_ratio(t_dl, t_sst)})"
    )

    results.append(
        BenchmarkResult(
            benchmark="remote_download_then_load",
            model="",
            metrics={"seconds": round(t_dl, 6)},
        )
    )
    results.append(
        BenchmarkResult(
            benchmark="remote_streaming_load",
            model="",
            metrics={
                "seconds": round(t_sst, 6),
                "ratio_vs_download": round(t_dl / t_sst, 4) if t_sst > 0 else 0,
            },
        )
    )
    return results


def bench_streaming_iteration(
    model_info: dict[str, str], runs: int
) -> list[BenchmarkResult]:
    """Measure stream_tensors() time-to-first-tensor and total iteration time."""
    import safetensors_streaming

    url = model_info["url"]
    results: list[BenchmarkResult] = []

    def stream_once() -> tuple[float, float]:
        start = time.perf_counter()
        first_tensor_time: float | None = None
        for _name, _tensor in safetensors_streaming.stream_tensors(url):
            if first_tensor_time is None:
                first_tensor_time = time.perf_counter() - start
        total = time.perf_counter() - start
        return first_tensor_time if first_tensor_time is not None else total, total

    # Collect runs
    first_times: list[float] = []
    total_times: list[float] = []
    for _ in range(runs):
        first, total = stream_once()
        first_times.append(first)
        total_times.append(total)

    t_first = statistics.median(first_times)
    t_total = statistics.median(total_times)

    print(f"\nStreaming iteration:")
    print(f"  Time to first tensor:               {format_time(t_first):>10}")
    print(f"  Total iteration time:              {format_time(t_total):>10}")

    results.append(
        BenchmarkResult(
            benchmark="streaming_iteration",
            model="",
            metrics={
                "first_tensor_seconds": round(t_first, 6),
                "total_seconds": round(t_total, 6),
            },
        )
    )
    return results


def bench_gpu_transfer(
    model_info: dict[str, str], local_path: str, runs: int
) -> list[BenchmarkResult]:
    """Compare download+load+.cuda() vs streaming load for GPU transfer."""
    import torch
    import safetensors.torch
    import safetensors_streaming

    url = model_info["url"]
    repo_id = model_info["hf_repo"]
    filename = model_info["hf_filename"]
    device = "cuda:0"
    results: list[BenchmarkResult] = []

    def time_download_load_cuda() -> float:
        with tempfile.TemporaryDirectory() as tmp:
            start = time.perf_counter()
            path = download_via_hf_hub(repo_id, filename, cache_dir=tmp)
            tensors = safetensors.torch.load_file(path)
            for key in tensors:
                tensors[key] = tensors[key].to(device)
            torch.cuda.synchronize()
            return time.perf_counter() - start

    def time_local_load_cuda() -> float:
        start = time.perf_counter()
        tensors = safetensors.torch.load_file(local_path)
        for key in tensors:
            tensors[key] = tensors[key].to(device)
        torch.cuda.synchronize()
        return time.perf_counter() - start

    def time_streaming_load() -> float:
        start = time.perf_counter()
        tensors = safetensors_streaming.load_file(url)
        # Currently load_file returns CPU tensors; move to GPU
        for key in tensors:
            tensors[key] = tensors[key].to(device)
        torch.cuda.synchronize()
        return time.perf_counter() - start

    t_dl_cuda = median_of(time_download_load_cuda, runs)
    t_local_cuda = median_of(time_local_load_cuda, runs)
    t_sst = median_of(time_streaming_load, runs)

    print(f"\nGPU transfer ({device}):")
    print(f"  download + load + .cuda():         {format_time(t_dl_cuda):>10}")
    print(f"  local load + .cuda():              {format_time(t_local_cuda):>10}")
    print(
        f"  streaming load(url) + .cuda():     {format_time(t_sst):>10}  ({format_ratio(t_dl_cuda, t_sst)})"
    )

    results.append(
        BenchmarkResult(
            benchmark="gpu_download_load_cuda",
            model="",
            metrics={"seconds": round(t_dl_cuda, 6)},
        )
    )
    results.append(
        BenchmarkResult(
            benchmark="gpu_local_load_cuda",
            model="",
            metrics={"seconds": round(t_local_cuda, 6)},
        )
    )
    results.append(
        BenchmarkResult(
            benchmark="gpu_streaming_load_cuda",
            model="",
            metrics={
                "seconds": round(t_sst, 6),
                "ratio_vs_download_cuda": round(t_dl_cuda / t_sst, 4) if t_sst > 0 else 0,
            },
        )
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_benchmarks_for_model(
    model_key: str,
    model_info: dict[str, str],
    *,
    skip_remote: bool,
    skip_gpu: bool,
    runs: int,
) -> BenchmarkRun | None:
    """Run all applicable benchmarks for one model and return results."""
    url = model_info["url"]
    name = model_info["name"]

    print(f"\n{'=' * 60}")
    print(f"Model: {name}")
    print(f"URL: {url}")

    # Download model to a temp location for local benchmarks
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, "model.safetensors")

        print(f"Downloading model to temp directory...")
        t_download = timed(lambda: download_model(url, local_path))
        file_size = os.path.getsize(local_path)
        print(
            f"Downloaded {file_size / 1e6:.1f} MB in {format_time(t_download)}"
        )

        # Count tensors
        import safetensors.torch

        tensors = safetensors.torch.load_file(local_path)
        tensor_count = len(tensors)
        del tensors  # free memory

        print(f"Model: {name} ({file_size / 1e6:.1f} MB, {tensor_count} tensors)")

        has_cuda = check_cuda()
        run_result = BenchmarkRun(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            model=name,
            file_size_bytes=file_size,
            tensor_count=tensor_count,
            cuda_available=has_cuda,
        )

        # 1. Local load benchmark
        run_result.results.extend(bench_local_load(local_path, file_size, runs))

        # 2. Remote load benchmark
        if not skip_remote:
            run_result.results.extend(bench_remote_load(model_info, runs))
        else:
            print("\nRemote load: SKIPPED (--skip-remote)")

        # 3. Streaming iteration
        if not skip_remote:
            run_result.results.extend(
                bench_streaming_iteration(model_info, runs)
            )
        else:
            print("\nStreaming iteration: SKIPPED (--skip-remote)")

        # 4. GPU transfer
        if not skip_gpu and has_cuda:
            run_result.results.extend(
                bench_gpu_transfer(model_info, local_path, runs)
            )
        elif skip_gpu:
            print("\nGPU transfer: SKIPPED (--skip-gpu)")
        else:
            print("\nGPU transfer: SKIPPED (no CUDA available)")

        # Tag all results with model name
        for r in run_result.results:
            r.model = name

    return run_result


def save_results(results: list[BenchmarkRun]) -> Path:
    """Save benchmark results as JSON and return the file path."""
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"bench_{timestamp}.json"
    data = [asdict(r) for r in results]
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="safetensors-streaming benchmark suite"
    )
    parser.add_argument(
        "--model",
        choices=["small", "medium", "all"],
        default="small",
        help="Which model(s) to benchmark (default: small)",
    )
    parser.add_argument(
        "--skip-remote",
        action="store_true",
        help="Skip remote/URL benchmarks",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip GPU transfer benchmarks",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per benchmark for median (default: 3)",
    )
    args = parser.parse_args()

    # Check dependencies
    missing = preflight()
    if missing:
        print("Missing required packages:", ", ".join(missing))
        print()
        print("Install them with:")
        if "safetensors_streaming" in missing:
            print("  # Build safetensors_streaming from source:")
            print("  cd crates && maturin develop --release -m sst-python/Cargo.toml")
        pip_deps = [p for p in missing if p != "safetensors_streaming"]
        if pip_deps:
            print(f"  pip install {' '.join(pip_deps)}")
        sys.exit(1)

    # Determine which models to run
    if args.model == "all":
        model_keys = list(MODELS.keys())
    else:
        model_keys = [args.model]

    print("=== safetensors-streaming benchmark ===")
    print(f"Runs per benchmark: {args.runs} (reporting median)")
    if check_cuda():
        import torch

        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA: not available")

    all_results: list[BenchmarkRun] = []

    for model_key in model_keys:
        result = run_benchmarks_for_model(
            model_key,
            MODELS[model_key],
            skip_remote=args.skip_remote,
            skip_gpu=args.skip_gpu,
            runs=args.runs,
        )
        if result is not None:
            all_results.append(result)

    # Save results
    if all_results:
        out_path = save_results(all_results)
        print(f"\n{'=' * 60}")
        print(f"Results saved to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
