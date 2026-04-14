#!/usr/bin/env python3
"""Verify A6000 environment setup for CUPS + DINOv3 pipeline."""

import sys
import os


def check(name, fn):
    """Run a check and print pass/fail."""
    try:
        result = fn()
        print(f"  [PASS] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def main():
    passed = 0
    failed = 0
    total_checks = []

    # ── Section 1: Python ──
    print("\n=== Python ===")
    total_checks.append(check(
        "Python version >= 3.11",
        lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 11) else (_ for _ in ()).throw(
            RuntimeError(f"Got {sys.version_info.major}.{sys.version_info.minor}"))
    ))
    total_checks.append(check(
        "Running inside venv",
        lambda: sys.prefix if sys.prefix != sys.base_prefix else (_ for _ in ()).throw(
            RuntimeError("Not in a virtual environment"))
    ))

    # ── Section 2: PyTorch + CUDA ──
    print("\n=== PyTorch + CUDA ===")
    total_checks.append(check(
        "torch import",
        lambda: __import__("torch").__version__
    ))
    total_checks.append(check(
        "CUDA available",
        lambda: (lambda t: f"Yes, {t.cuda.device_count()} device(s)"
                 if t.cuda.is_available() else (_ for _ in ()).throw(
                     RuntimeError("CUDA not available")))(__import__("torch"))
    ))
    total_checks.append(check(
        "GPU name",
        lambda: __import__("torch").cuda.get_device_name(0)
    ))
    total_checks.append(check(
        "CUDA version (torch)",
        lambda: __import__("torch").version.cuda
    ))
    total_checks.append(check(
        "cuDNN version",
        lambda: str(__import__("torch").backends.cudnn.version())
    ))
    total_checks.append(check(
        "GPU memory",
        lambda: f"{__import__('torch').cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    ))
    total_checks.append(check(
        "torch.compile available",
        lambda: "Yes" if hasattr(__import__("torch"), "compile") else "No"
    ))

    # ── Section 3: Detectron2 ──
    print("\n=== Detectron2 ===")
    total_checks.append(check(
        "detectron2 import",
        lambda: __import__("detectron2").__version__
    ))
    def _check_d2_cuda_ops():
        from detectron2 import _C  # noqa: F401
        return "OK"
    total_checks.append(check("detectron2 CUDA ops", _check_d2_cuda_ops))

    def _check_d2_model_zoo():
        from detectron2 import model_zoo  # noqa: F401
        return "OK"
    total_checks.append(check("detectron2.model_zoo", _check_d2_model_zoo))

    # ── Section 4: DINOv3 ──
    print("\n=== DINOv3 ===")
    total_checks.append(check(
        "dinov3 import",
        lambda: "OK" if __import__("dinov3") else "OK"
    ))

    # ── Section 5: CUPS core dependencies ──
    print("\n=== CUPS Dependencies ===")
    cups_deps = [
        ("kornia", None),
        ("PIL", "Pillow"),
        ("scipy", None),
        ("yacs", None),
        ("wandb", None),
        ("pytorch_lightning", "pytorch-lightning"),
        ("timm", None),
        ("pycocotools", None),
        ("sklearn", "scikit-learn"),
        ("torchmetrics", None),
        ("skimage", "scikit-image"),
        ("optuna", None),
        ("einops", None),
        ("cv2", "opencv-python"),
        ("faster_coco_eval", "faster-coco-eval"),
    ]
    for module_name, display_name in cups_deps:
        name = display_name or module_name
        total_checks.append(check(
            name,
            lambda m=module_name: getattr(__import__(m), "__version__", "OK")
        ))

    # ── Section 6: DINOv3 dependencies ──
    print("\n=== DINOv3 Dependencies ===")
    dinov3_deps = [
        ("ftfy", None),
        ("omegaconf", None),
        ("regex", None),
        ("termcolor", None),
        ("torchvision", None),
    ]
    for module_name, display_name in dinov3_deps:
        name = display_name or module_name
        total_checks.append(check(
            name,
            lambda m=module_name: getattr(__import__(m), "__version__", "OK")
        ))

    # ── Section 7: Environment variables ──
    print("\n=== Environment ===")
    total_checks.append(check(
        "CUDA_HOME set",
        lambda: os.environ.get("CUDA_HOME", None) or (_ for _ in ()).throw(
            RuntimeError("CUDA_HOME not set"))
    ))
    total_checks.append(check(
        "nvcc accessible",
        lambda: (lambda r: r if r == 0 else (_ for _ in ()).throw(
            RuntimeError("nvcc not found in PATH")))(os.system("nvcc --version > /dev/null 2>&1"))
    ))

    # ── Section 8: Quick GPU smoke test ──
    print("\n=== GPU Smoke Test ===")
    total_checks.append(check(
        "Tensor to GPU and back",
        lambda: (lambda t: f"OK ({t.cuda.FloatTensor(2, 3).cpu().shape})")(
            __import__("torch"))
    ))
    total_checks.append(check(
        "cuDNN convolution",
        lambda: (lambda t: (
            t.nn.functional.conv2d(
                t.randn(1, 3, 8, 8, device="cuda"),
                t.randn(16, 3, 3, 3, device="cuda")
            ).shape, "OK")[1])(__import__("torch"))
    ))

    # ── Summary ──
    passed = sum(1 for c in total_checks if c)
    failed = sum(1 for c in total_checks if not c)
    total = len(total_checks)

    print(f"\n{'='*50}")
    print(f"  PASSED: {passed}/{total}")
    if failed > 0:
        print(f"  FAILED: {failed}/{total}")
        print(f"\nFix the failures above before running training.")
        sys.exit(1)
    else:
        print(f"\n  Environment is ready for CUPS + DINOv3 training!")
        sys.exit(0)


if __name__ == "__main__":
    main()
