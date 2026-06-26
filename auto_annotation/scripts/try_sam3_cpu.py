#!/usr/bin/env python3
"""Empirical probe: can SAM 3 IMAGE inference run on CPU (macOS, no CUDA)?

Strategy (no edits to vendored sam3 source):
  1. Inject a stub `triton` module so import-time @triton.jit/@autotune decorators in
     edt.py / connected_components.py don't crash (those kernels are video-tracker only
     and are never CALLED in the text-prompt image path).
  2. Monkeypatch torch so hardcoded device="cuda" tensors / .cuda() / .to("cuda") /
     autocast("cuda") coerce to cpu, and torch.compile (needs triton) becomes a no-op.
  3. build_sam3_image_model(device="cpu") -> Sam3Processor(device="cpu") -> one image,
     text prompt "car". Report timing + detections, stage by stage.

Run: .venv_cups_cpu/bin/python auto_annotation/scripts/try_sam3_cpu.py
"""

import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# ---------- 1) triton stub (only decorators run at import; kernel bodies never called) ----------
class _Any:
    """Permissive sentinel: callable, subscriptable, any-attr — for unused kernels."""
    def __call__(self, *a, **k): return _ANY
    def __getitem__(self, i): return _ANY
    def __getattr__(self, n):
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)
        return _ANY


_ANY = _Any()


class _PermissiveModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):  # let inspect/import see real dunders
            raise AttributeError(name)
        return _ANY


def _install_triton_stub() -> None:
    """Serve a permissive package for ANY `triton.*` import depth via a meta finder."""
    import importlib.abc
    import importlib.machinery

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path, target=None):
            if name == "triton" or name.startswith("triton."):
                return importlib.machinery.ModuleSpec(name, self)
            return None

        def create_module(self, spec):
            m = _PermissiveModule(spec.name)
            m.__path__ = []  # mark as a package so submodule imports resolve
            if spec.name == "triton":
                m.jit = lambda fn=None, **k: (fn if callable(fn) else (lambda f: f))
                m.autotune = lambda *a, **k: (lambda f: f)
                m.heuristics = lambda *a, **k: (lambda f: f)
                m.Config = lambda *a, **k: None
                m.cdiv = lambda a, b: (a + b - 1) // b
                m.__version__ = "3.1.0-stub"
            if spec.name == "triton.language":
                m.constexpr = _ANY
            return m

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _Finder())


def _install_cuda_to_cpu_shims() -> None:
    import torch

    def coerce(d):
        return "cpu" if (d is not None and str(d).startswith("cuda")) else d

    for name in ("zeros", "ones", "empty", "full", "tensor", "arange", "randn", "rand",
                 "as_tensor", "eye"):
        orig = getattr(torch, name)

        def make(orig):
            def g(*a, **k):
                if "device" in k:
                    k["device"] = coerce(k["device"])
                return orig(*a, **k)
            return g
        setattr(torch, name, make(orig))

    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.Tensor.pin_memory = lambda self, *a, **k: self  # CUDA-only; no-op on CPU
    import torch.nn as nn
    nn.Module.cuda = lambda self, *a, **k: self

    # force fp32 everywhere: CPU has no bf16/fp16 kernels for many ops, and the model
    # casts activations to bf16 (expecting CUDA autocast) while weights stay fp32.
    _lowp = (torch.bfloat16, torch.float16)
    torch.Tensor.bfloat16 = lambda self, *a, **k: self
    torch.Tensor.half = lambda self, *a, **k: self
    nn.Module.bfloat16 = lambda self, *a, **k: self
    nn.Module.half = lambda self, *a, **k: self

    orig_to = torch.Tensor.to

    def to(self, *a, **k):
        a = tuple(coerce(x) if isinstance(x, str) else x for x in a)
        a = tuple(x for x in a if not (isinstance(x, torch.dtype) and x in _lowp))
        if k.get("dtype") in _lowp:
            k.pop("dtype")
        if "device" in k:
            k["device"] = coerce(k["device"])
        return orig_to(self, *a, **k)
    torch.Tensor.to = to

    orig_mto = nn.Module.to

    def mto(self, *a, **k):
        a = tuple(coerce(x) if isinstance(x, str) else x for x in a)
        a = tuple(x for x in a if not (isinstance(x, torch.dtype) and x in _lowp))
        if k.get("dtype") in _lowp:
            k.pop("dtype")
        if "device" in k:
            k["device"] = coerce(k["device"])
        return orig_mto(self, *a, **k)
    nn.Module.to = mto

    orig_autocast = torch.autocast

    def autocast(device_type="cpu", *a, **k):  # keep ContextDecorator behavior
        return orig_autocast(coerce(device_type) or "cpu", enabled=False)
    torch.autocast = autocast
    try:
        torch.cuda.amp.autocast = lambda *a, **k: orig_autocast("cpu", enabled=False)
    except Exception:
        pass

    torch.compile = lambda m=None, *a, **k: (m if m is not None else (lambda f: f))


def main() -> None:
    t0 = time.time()
    _install_cuda_to_cpu_shims()  # imports torch fully BEFORE the triton stub exists
    _install_triton_stub()
    sys.path.insert(0, str(ROOT / "external" / "sam3"))

    import numpy as np
    from PIL import Image

    print(f"[{time.time()-t0:.0f}s] stubs installed; importing sam3 ...")
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as e:
        print("IMPORT FAILED:", type(e).__name__, str(e)[:300]); return

    print(f"[{time.time()-t0:.0f}s] import OK; building model on CPU "
          f"(downloads sam3.pt ~3.45GB first run) ...")
    try:
        model = build_sam3_image_model(device="cpu")
    except Exception as e:
        import traceback; traceback.print_exc()
        print("BUILD FAILED:", type(e).__name__, str(e)[:300]); return

    print(f"[{time.time()-t0:.0f}s] model built; preparing processor + image ...")
    proc = Sam3Processor(model, resolution=1008, device="cpu")  # canonical res (RoPE table)

    img_path = sorted(Path("/Volumes/code_files/datasets/cityscapes/leftImg8bit/val/"
                            "frankfurt").glob("*_leftImg8bit.png"))[2]
    image = Image.open(img_path).convert("RGB")
    print(f"[{time.time()-t0:.0f}s] image: {img_path.name} {image.size}; running "
          f"set_image + text prompt 'car' ...")
    try:
        t1 = time.time()
        state = proc.set_image(image)
        out = proc.set_text_prompt(state=state, prompt="car")
        dt = time.time() - t1
    except Exception as e:
        import traceback; traceback.print_exc()
        print("INFERENCE FAILED:", type(e).__name__, str(e)[:300]); return

    masks, boxes, scores = out["masks"], out["boxes"], out["scores"]
    n = 0 if masks is None else (len(masks) if hasattr(masks, "__len__") else masks.shape[0])
    print(f"\n[SUCCESS] SAM3 ran on CPU. inference={dt:.1f}s | "
          f"detections('car')={n}")
    try:
        sc = [round(float(s), 3) for s in (scores if scores is not None else [])][:8]
        print(f"  top scores: {sc}")
        print(f"  masks shape: {tuple(masks.shape) if hasattr(masks,'shape') else type(masks)}")
    except Exception:
        pass

    # save a quick overlay
    try:
        out_dir = ROOT / "auto_annotation/outputs/sam3_cpu"
        out_dir.mkdir(parents=True, exist_ok=True)
        base = np.array(image).copy()
        m = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
        if m.ndim == 4:
            m = m[:, 0]
        for i in range(min(n, 20)):
            mm = m[i] > 0.5
            if mm.shape != base.shape[:2]:
                mm = np.array(Image.fromarray(mm.astype(np.uint8)).resize(
                    image.size, Image.NEAREST)).astype(bool)
            base[mm] = (0.5 * base[mm] + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        Image.fromarray(base).save(out_dir / f"{img_path.stem}_sam3_car.png")
        print(f"  overlay -> {out_dir}/{img_path.stem}_sam3_car.png")
    except Exception as e:
        print("  (overlay skipped:", str(e)[:80], ")")

    print(f"[{time.time()-t0:.0f}s] done.")


if __name__ == "__main__":
    main()
