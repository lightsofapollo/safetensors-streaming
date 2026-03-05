# gpu-cli bugs encountered during safetensors-streaming development

## 1. Disk quota exceeded on network volume
- **What**: `aws-lc-sys` (AWS SDK dep) compiles BoringSSL from C source, generating GBs of build artifacts. Filled up network volume causing `Disk quota exceeded (os error 122)`.
- **Root cause**: RunPod network volume assigned to a datacenter with low quota. Previous failed compiles left orphaned artifacts.
- **Workaround**: Deleted the old volume in RunPod dashboard, created new one. Also added `--release` to cargo builds (smaller artifacts than debug).

## 2. Shell steps reference wrong cargo path
- **What**: `gpu.jsonc` shell steps using `$HOME/.cargo/bin/cargo` fail because gpu-cli sets `CARGO_HOME=/gpu-cli-workspaces/.cache/cargo/`.
- **Workaround**: Use `source /gpu-cli-workspaces/.cache/cargo/env` in run commands instead of `source ~/.cargo/env`.
- **Suggestion**: Document the actual cargo path in gpu-cli docs, or make `~/.cargo/env` a symlink.

## 3. `cargo install cargo-nextest` too slow / needs --locked
- **What**: Compiling cargo-nextest from source takes 5+ minutes and the `locked-tripwire` crate enforces `--locked` flag. Connection drops during long compiles.
- **Workaround**: Use pre-built binary: `curl -LsSf https://get.nexte.st/latest/linux | tar zxf - --no-same-owner -C /gpu-cli-workspaces/.cache/cargo/bin`
- **Suggestion**: Support pre-built binary installs in `gpu.jsonc` config.

## 4. `cargo install maturin` fails due to yanked dependency
- **What**: `maturin v1.12.6` depends on `cargo-xwin` which depends on `xwin ^0.6.6`, but versions 0.6.6 and 0.6.7 are yanked on crates.io.
- **Workaround**: Install via pip instead: `pip install maturin` (pre-built binary, much faster).

## 5. tar ownership error when extracting pre-built binaries
- **What**: `tar zxf` fails with "Cannot change ownership to uid 1001" when extracting nextest binary.
- **Workaround**: Add `--no-same-owner` flag to tar.

## 6. .venv pip permissions issue
- **What**: `pip install` into the `.venv` created by gpu-cli fails with "Permission denied" for `.venv/bin/pip`.
- **Workaround**: Install packages system-wide with `pip install` in the run command instead of relying on requirements-dev.txt.
- **Suggestion**: Fix .venv creation permissions or don't use .venv on pods.

## 7. Shell setup re-runs on every `gpu run`
- **What**: All shell steps (rustup, cargo install, pip install) re-run on every `gpu run` invocation even when the pod is warm and tools are already installed.
- **Impact**: Adds ~15-20s of overhead per command. Rustup alone downloads the installer each time.
- **Suggestion**: Cache a "setup complete" marker and skip if present, or move setup to pod creation only.
