#!/usr/bin/env bash
set -euo pipefail

# Build script for safetensors-streaming wheels.
#
# Produces two wheels:
#   1. safetensors-streaming       — CPU-only (no CUDA dependency)
#   2. safetensors-streaming-cu12  — with CUDA 12.x support
#
# The CUDA wheel requires the CUDA 12.x toolkit to be installed on the
# build machine (nvcc, libcuda, etc.). On machines without CUDA, only
# the CPU wheel will be built.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Building CPU wheel (safetensors-streaming) ==="
# Default build: no extra features, produces a pure CPU wheel.
maturin build --release --manifest-path "$PROJECT_ROOT/crates/sst-python/Cargo.toml"

echo ""
echo "=== Building CUDA 12.x wheel (safetensors-streaming-cu12) ==="
# Enables the 'cuda' feature which pulls in sst-gpu/cuda (cudarc bindings).
# This requires the CUDA toolkit on the build machine.
# After building, the wheel should be renamed/published as safetensors-streaming-cu12
# to distinguish it from the CPU-only package.
maturin build --release --manifest-path "$PROJECT_ROOT/crates/sst-python/Cargo.toml" --features cuda

echo ""
echo "=== Done ==="
echo "Wheels are in $PROJECT_ROOT/target/wheels/"
