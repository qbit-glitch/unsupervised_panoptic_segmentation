# Stage-1 LoRA Adapter Implementation — File Locations

This document catalogs all files involved in the **stage-1 frozen-backbone LoRA/DoRA adapter** pipeline for MBPS.

---

## 1. Core Layer Implementations

| File | Purpose |
|------|---------|
| [`mbps_pytorch/models/adapters/lora_layers.py`](../mbps_pytorch/models/adapters/lora_layers.py) | Core adapter layers: `LoRALinear`, `DoRALinear`, `ConvDoRALinear`, `LoRAConv2d`, plus `freeze_non_adapter_params()`, merge/unmerge helpers, and parameter-count utilities. |
| [`mbps_pytorch/models/adapters/__init__.py`](../mbps_pytorch/models/adapters/__init__.py) | Package-level exports. Imports all layer classes and backbone-specific injection functions. |

---

## 2. Backbone-Specific Adapter Injection

These modules contain the functions that **wrap** target linear/Conv2d layers inside pretrained backbones with the LoRA/DoRA adapters.

| File | Target Backbone / Head | Key Function |
|------|------------------------|--------------|
| [`mbps_pytorch/models/adapters/dinov2_adapter.py`](../mbps_pytorch/models/adapters/dinov2_adapter.py) | DINOv2 ViT backbone | `inject_lora_into_dinov2()` |
| [`mbps_pytorch/models/adapters/cause_adapter.py`](../mbps_pytorch/models/adapters/cause_adapter.py) | CAUSE-TR `Segment_TR` head | `inject_lora_into_cause_tr()` |
| [`mbps_pytorch/models/adapters/depth_adapter.py`](../mbps_pytorch/models/adapters/depth_adapter.py) | Depth Anything V3 encoder | `inject_lora_into_depth_model()` |
| [`mbps_pytorch/models/adapters/depthpro_adapter.py`](../mbps_pytorch/models/adapters/depthpro_adapter.py) | DepthPro encoder | `inject_lora_into_depthpro()` |

---

## 3. Stage-1 Training Scripts (Frozen Backbone)

These are the **entry-point scripts** that load a pretrained backbone, inject adapters, freeze all non-adapter parameters, and run self-supervised training.

| File | What It Trains |
|------|----------------|
| [`mbps_pytorch/train_semantic_adapter.py`](../mbps_pytorch/train_semantic_adapter.py) | Self-supervised adapter training on **DINOv2 + CAUSE-TR**. The backbone and EMA teacher are frozen; only LoRA/DoRA parameters are updated. |
| [`mbps_pytorch/train_depth_adapter_lora.py`](../mbps_pytorch/train_depth_adapter_lora.py) | Self-supervised adapter training on **monocular depth models** (DAv3 / DepthPro). The depth encoder/decoder is frozen; only adapter parameters train. |

---

## 4. Baseline Configurations

YAML config files used to launch stage-1 adapter training jobs.

| File |
|------|
| [`configs/semantic_adapter_baseline.yaml`](../configs/semantic_adapter_baseline.yaml) |
| [`configs/depth_adapter_baseline.yaml`](../configs/depth_adapter_baseline.yaml) |
| [`configs/depthpro_adapter_baseline.yaml`](../configs/depthpro_adapter_baseline.yaml) |

---

## 5. Tests / Smoke Tests

| File | Scope |
|------|-------|
| [`mbps_pytorch/tests/test_adapters.py`](../mbps_pytorch/tests/test_adapters.py) | Unit tests for layer correctness and merge/unmerge reversibility. |
| [`mbps_pytorch/tests/smoke_test_semantic_adapter.py`](../mbps_pytorch/tests/smoke_test_semantic_adapter.py) | End-to-end smoke test for semantic adapter training loop. |
| [`mbps_pytorch/tests/smoke_test_depth_adapter.py`](../mbps_pytorch/tests/smoke_test_depth_adapter.py) | End-to-end smoke test for depth adapter training loop. |

---

## 6. Key Design Pattern — Freezing the Backbone

The frozen-backbone guarantee is enforced by [`freeze_non_adapter_params()`](mbps_pytorch/models/adapters/lora_layers.py) (line 389 in `lora_layers.py`):

```python
def freeze_non_adapter_params(model: nn.Module) -> None:
    ADAPTER_SUFFIXES = (
        ".lora_A", ".lora_B", ".lora_magnitude",
        ".dwconv.weight", ".conv_gate",
        ".lora_A.weight", ".lora_B.weight"
    )
    for name, param in model.named_parameters():
        if any(name.endswith(suffix) for suffix in ADAPTER_SUFFIXES):
            param.requires_grad = True
        else:
            param.requires_grad = False
```

This is called **after** the injection functions have replaced target layers with their adapter-wrapped counterparts, ensuring only the low-rank residuals (and optional DWConv gates) receive gradients.

---

*Generated: 2026-04-25*
