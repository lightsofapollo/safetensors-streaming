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

## 8. pyproject.toml auto-detected triggers unwanted maturin build
- **What**: gpu-cli auto-detects `pyproject.toml` and creates a `.venv`, then builds the project from source using maturin. This happens even when: (a) the `python` section is removed from `gpu.jsonc`, (b) the prebuilt wheel is already installed to system python via shell steps.
- **Impact**: 5+ minutes of Rust compilation on every fresh pod, completely defeating the purpose of cross-compiled prebuilt wheels. The relay session can expire during this build, killing the entire job.
- **Root cause**: gpu-cli unconditionally scans for pyproject.toml and treats it as an installable project. There's no way to opt out.
- **Suggestion**: Add a config option like `"python": false` or `"skip_pyproject": true` to disable auto-detection. Or respect the absence of the `python` section in gpu.jsonc as "don't touch python packaging". Projects that cross-compile wheels locally should not be forced to rebuild on every pod.

## 9. No `.gpuignore` mechanism to exclude files from sync
- **What**: All project files (minus .gitignore patterns) are synced to the pod. There's no way to exclude files that shouldn't be on the pod (e.g., `pyproject.toml` when using prebuilt wheels, or large test fixtures).
- **Suggestion**: Support a `.gpuignore` file (same syntax as .gitignore) for pod-specific exclusions. This would let projects exclude `pyproject.toml` to prevent the maturin auto-build issue (#8).

## 10. Relay session expires during long dependency installs
- **What**: When dependency installation (maturin build, pip install vllm) takes more than ~10 minutes, the relay session expires with "Relay handshake failed: Session expired", killing the entire job.
- **Impact**: Large packages like vLLM (~2GB+ of wheel downloads + compilation) routinely exceed this timeout.
- **Suggestion**: The relay session timeout should be much longer during dependency installation, or the session should be refreshed/kept alive while deps are being installed. Alternatively, dependency installation could happen asynchronously with the relay reconnecting when done.

## 11. `gpu run` command uses .venv python, not system python
- **What**: Commands submitted via `gpu run` execute inside the auto-created `.venv`. Even if the prebuilt wheel is installed to system python (via shell steps), the `gpu run` command uses the venv's python which may have a different (maturin-built) version.
- **Impact**: No way to use `/usr/bin/python` directly — the venv activation is implicit. Using `/usr/bin/python` in the command works but feels like a hack.
- **Suggestion**: Let users control which python is used, or at least document this behavior. Consider `"use_venv": false` in gpu.jsonc.
