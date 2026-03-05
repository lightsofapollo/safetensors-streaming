"""Integration tests for safetensors_streaming patch/patch_hub with the transformers library.

Run via:
    gpu run "source /gpu-cli-workspaces/.cache/cargo/env && maturin develop --release && python3 tests/test_transformers.py"
"""

from __future__ import annotations

import sys
import time
import traceback

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"

passed = 0
failed = 0


def make_dummy_input() -> dict[str, torch.Tensor]:
    """Create dummy tokenizer input for the test model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer("Hello world, this is a test.", return_tensors="pt")
    return inputs


def run_forward(model: AutoModel, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run a forward pass and return the last hidden state."""
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state


def report(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    status = "PASS" if ok else "FAIL"
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if ok:
        passed += 1
    else:
        failed += 1


# ── Prepare inputs once ────────────────────────────────────────────────

print(f"Model: {MODEL_ID}")
print("Preparing tokenizer and dummy inputs...")
inputs = make_dummy_input()
print(f"Input shape: {inputs['input_ids'].shape}")
print()


# ── Test 0: Baseline (no patching) ─────────────────────────────────────

print("=" * 60)
print("Test 0: Baseline — load model without any patching")
print("=" * 60)

baseline_output: torch.Tensor | None = None
try:
    t0 = time.monotonic()
    model_baseline = AutoModel.from_pretrained(MODEL_ID)
    model_baseline.eval()
    baseline_output = run_forward(model_baseline, inputs)
    elapsed = time.monotonic() - t0
    report("baseline load + forward", True, f"output shape={baseline_output.shape}, time={elapsed:.2f}s")
    del model_baseline
except Exception:
    traceback.print_exc()
    report("baseline load + forward", False, "exception during baseline")

print()


# ── Test 1: patch() / unpatch() ────────────────────────────────────────

print("=" * 60)
print("Test 1: patch() — monkey-patch safetensors, load model, run forward")
print("=" * 60)

patch_output: torch.Tensor | None = None
try:
    import safetensors_streaming

    safetensors_streaming.patch()
    print("  patch() applied")

    t0 = time.monotonic()
    model_patched = AutoModel.from_pretrained(MODEL_ID)
    model_patched.eval()
    patch_output = run_forward(model_patched, inputs)
    elapsed = time.monotonic() - t0

    report("patch load + forward", True, f"output shape={patch_output.shape}, time={elapsed:.2f}s")
    del model_patched

    safetensors_streaming.unpatch()
    print("  unpatch() called")
except Exception:
    traceback.print_exc()
    report("patch load + forward", False, "exception during patched load")
    # Try to unpatch in case it was applied
    try:
        safetensors_streaming.unpatch()
    except Exception:
        pass

print()


# ── Test 2: patch_hub() / unpatch_hub() ────────────────────────────────

print("=" * 60)
print("Test 2: patch_hub() — intercept hf_hub_download for .safetensors, load model")
print("=" * 60)

hub_output: torch.Tensor | None = None
try:
    from safetensors_streaming.hub import patch_hub, unpatch_hub

    # patch() is also needed so that safetensors.torch.load_file can handle URLs
    safetensors_streaming.patch()
    patch_hub()
    print("  patch() + patch_hub() applied")

    t0 = time.monotonic()
    model_hub = AutoModel.from_pretrained(MODEL_ID)
    model_hub.eval()
    hub_output = run_forward(model_hub, inputs)
    elapsed = time.monotonic() - t0

    report("patch_hub load + forward", True, f"output shape={hub_output.shape}, time={elapsed:.2f}s")
    del model_hub

    unpatch_hub()
    safetensors_streaming.unpatch()
    print("  unpatch_hub() + unpatch() called")
except Exception:
    traceback.print_exc()
    report("patch_hub load + forward", False, "exception during hub-patched load")
    try:
        unpatch_hub()
    except Exception:
        pass
    try:
        safetensors_streaming.unpatch()
    except Exception:
        pass

print()


# ── Test 3: patched() context manager ──────────────────────────────────

print("=" * 60)
print("Test 3: patched() context manager — scoped monkey-patch")
print("=" * 60)

ctx_output: torch.Tensor | None = None
try:
    with safetensors_streaming.patched():
        print("  inside patched() context")
        t0 = time.monotonic()
        model_ctx = AutoModel.from_pretrained(MODEL_ID)
        model_ctx.eval()
        ctx_output = run_forward(model_ctx, inputs)
        elapsed = time.monotonic() - t0
        report("patched() load + forward", True, f"output shape={ctx_output.shape}, time={elapsed:.2f}s")
        del model_ctx

    # Verify unpatch happened automatically
    import safetensors
    import safetensors.torch

    from safetensors_streaming.safetensors_streaming import safe_open as sst_safe_open

    is_restored = safetensors.safe_open is not sst_safe_open
    report("patched() auto-unpatch", is_restored, "safetensors.safe_open restored after context exit")
except Exception:
    traceback.print_exc()
    report("patched() load + forward", False, "exception during context-managed load")

print()


# ── Test 4: Compare outputs ───────────────────────────────────────────

print("=" * 60)
print("Test 4: Compare outputs — all methods should produce identical results")
print("=" * 60)

try:
    if baseline_output is not None and patch_output is not None:
        match = torch.allclose(baseline_output, patch_output, atol=1e-6)
        report("baseline vs patch()", match, f"allclose={match}")
    else:
        report("baseline vs patch()", False, "one or both outputs missing")

    if baseline_output is not None and hub_output is not None:
        match = torch.allclose(baseline_output, hub_output, atol=1e-6)
        report("baseline vs patch_hub()", match, f"allclose={match}")
    else:
        report("baseline vs patch_hub()", False, "one or both outputs missing")

    if baseline_output is not None and ctx_output is not None:
        match = torch.allclose(baseline_output, ctx_output, atol=1e-6)
        report("baseline vs patched()", match, f"allclose={match}")
    else:
        report("baseline vs patched()", False, "one or both outputs missing")
except Exception:
    traceback.print_exc()
    report("output comparison", False, "exception during comparison")

print()


# ── Summary ────────────────────────────────────────────────────────────

print("=" * 60)
total = passed + failed
print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
print("=" * 60)

sys.exit(1 if failed > 0 else 0)
