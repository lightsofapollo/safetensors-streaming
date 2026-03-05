# CLAUDE.md

## What is safetensors-streaming

Rust+PyO3 library that streams safetensors files from HTTP/S3 directly to GPU memory. Drop-in replacement for `safetensors.safe_open()` and `safetensors.torch.load_file()` that accepts URLs.

## Build & Development

All Rust commands run from `crates/` (workspace root):

```bash
cd crates && cargo check --workspace
cd crates && cargo nextest run --workspace
cd crates && cargo clippy --workspace -- -D warnings
cd crates && cargo fmt --workspace
```

CLI benchmark:
```bash
cd crates && cargo run -p sst-cli -- fetch <URL> --benchmark
```

## GPU Testing (macOS)

This project requires NVIDIA GPUs for CUDA testing. Since macOS has no NVIDIA GPUs, use `gpu-cli` to test on remote GPU pods:

```bash
# Run Rust tests on GPU pod
gpu run "source ~/.cargo/env && cd crates && cargo nextest run --workspace"

# Build Python wheel with CUDA support on pod
gpu run "source ~/.cargo/env && maturin develop --release"

# Run Python tests on GPU
gpu run "pytest tests/"

# Run benchmarks on GPU
gpu run "python benches/benchmark.py"
```

The `gpu.jsonc` config provisions an RTX 4090 with Rust toolchain, maturin, and Python deps. Source code syncs to pod (target/ is gitignored — compile on pod). Results sync back via `outputs`.

**Do NOT attempt CUDA tests locally on macOS.** Always use `gpu run` for anything that needs a GPU.

## Workspace Crates

| Crate | Purpose |
|-------|---------|
| `sst-types` | Shared types: DType, error types |
| `sst-core` | Header parsing, pipeline orchestration |
| `sst-fetch` | HTTP/S3 async Range request fetcher |
| `sst-buffer` | Ring buffer (producer/consumer channel) |
| `sst-gpu` | CUDA pinned memory + async GPU copy (stub) |
| `sst-python` | PyO3 bindings (stub) |
| `sst-cli` | CLI benchmark tool |

## Coding Conventions

- **Edition 2024** for all crates
- **No `unwrap()` or `expect()`** in production code — use `?`, `thiserror`, `match`
- **No `any` type in TypeScript** — use proper types
- **No `println!`/`eprintln!`** — use `tracing` (except CLI user-facing output)
- Error handling: `thiserror` for library errors
- Do not use `window.confirm`, `alert`, etc. — use proper in-app UI
- Never sleep when you can wait on a condition

## Planning & Tracking

- Planning docs: `../private-gpu-ideas/planning/safetensors-streaming/`
- Issue tracking: Beads (`bd list`, `bd show`, `bd ready`)
- Skills: `plan-to-beads`, `implement-bd`

## Key Architecture Decisions

- **safe_open() is the primary API** — most frameworks use it, not load_file()
- **Dual-write cache**: stream to GPU AND disk simultaneously on first load
- **Two-tier HF support**: LFS bridge (Range requests) for now, Xet-native via cas_client later
- **Ring buffer pattern**: bounded, backpressure, async — swappable between heap and pinned CUDA memory
