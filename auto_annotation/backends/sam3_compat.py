#!/usr/bin/env python3
"""Make SAM 3 runnable on CPU (macOS / no-CUDA).

SAM 3 ships CUDA-coupled: a hard `import triton`, hardcoded device="cuda" tensors,
bf16 activations (expecting CUDA autocast), and `.pin_memory()`. None of that is
fundamental to inference — `enable_cpu_sam3()` installs shims so the IMAGE path runs
on CPU and produces correct masks (verified: text 'car' on Cityscapes -> 5 masks,
scores ~0.95, ~3s/img). Call it ONCE before importing sam3. CUDA is still preferred
for dataset-scale throughput.
"""

import sys
import types

__all__ = ["enable_cpu_sam3"]

_INSTALLED = False


class _Any:
    def __call__(self, *a, **k): return _ANY
    def __getitem__(self, i): return _ANY
    def __getattr__(self, n):
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)
        return _ANY


_ANY = _Any()


class _PermissiveModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _ANY


def _install_triton_stub() -> None:
    import importlib.abc
    import importlib.machinery

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path, target=None):
            if name == "triton" or name.startswith("triton."):
                return importlib.machinery.ModuleSpec(name, self)
            return None

        def create_module(self, spec):
            m = _PermissiveModule(spec.name)
            m.__path__ = []
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

    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


def _install_cuda_to_cpu_shims() -> None:
    import torch
    import torch.nn as nn

    def coerce(d):
        return "cpu" if (d is not None and str(d).startswith("cuda")) else d

    for name in ("zeros", "ones", "empty", "full", "tensor", "arange", "randn",
                 "rand", "as_tensor", "eye"):
        orig = getattr(torch, name)

        def make(orig):
            def g(*a, **k):
                if "device" in k:
                    k["device"] = coerce(k["device"])
                return orig(*a, **k)
            return g
        setattr(torch, name, make(orig))

    _lowp = (torch.bfloat16, torch.float16)
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    torch.Tensor.bfloat16 = lambda self, *a, **k: self
    torch.Tensor.half = lambda self, *a, **k: self
    nn.Module.cuda = lambda self, *a, **k: self
    nn.Module.bfloat16 = lambda self, *a, **k: self
    nn.Module.half = lambda self, *a, **k: self

    def _strip(a, k):
        a = tuple(coerce(x) if isinstance(x, str) else x for x in a)
        a = tuple(x for x in a if not (isinstance(x, torch.dtype) and x in _lowp))
        if k.get("dtype") in _lowp:
            k.pop("dtype")
        if "device" in k:
            k["device"] = coerce(k["device"])
        return a, k

    orig_to = torch.Tensor.to

    def t_to(self, *a, **k):
        a, k = _strip(a, k)
        return orig_to(self, *a, **k)
    torch.Tensor.to = t_to

    orig_mto = nn.Module.to

    def m_to(self, *a, **k):
        a, k = _strip(a, k)
        return orig_mto(self, *a, **k)
    nn.Module.to = m_to

    orig_autocast = torch.autocast
    torch.autocast = lambda device_type="cpu", *a, **k: orig_autocast(
        coerce(device_type) or "cpu", enabled=False)
    try:
        torch.cuda.amp.autocast = lambda *a, **k: orig_autocast("cpu", enabled=False)
    except Exception:
        pass
    torch.compile = lambda m=None, *a, **k: (m if m is not None else (lambda f: f))


def enable_cpu_sam3() -> None:
    """Install all CPU-compat shims (idempotent). Call before importing sam3."""
    global _INSTALLED
    if _INSTALLED:
        return
    _install_cuda_to_cpu_shims()  # import torch fully BEFORE the triton stub
    _install_triton_stub()
    _INSTALLED = True
