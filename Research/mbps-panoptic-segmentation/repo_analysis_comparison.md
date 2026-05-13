# Deep Repo Analysis: Comparing A* Paper Implementations with MBPS Stage-1 Adapters

**Analysis Date:** 2026-04-24
**Repos Analyzed:** ExPLoRA, GDA/SLR, LoRA-TTT, Uni-UVPT
**MBPS Codebase:** `mbps_pytorch/models/adapters/`

---

## 1. ExPLoRA (ICML 2025) — DINOv2/MAE + LoRA Self-Distillation

**Repo:** `github.com/samar-khanna/ExPLoRA`

### Adapter Architecture
```python
# Core LoRA layer: dinov2/layers/lora_layers_util.py
class LoRALinearLayer(nn.Linear, LoRALayer):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1, ...):
        # Standard LoRA: W' = W + (B @ A) * scaling
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.scaling = self.lora_alpha / self.r
        self.weight.requires_grad = False  # Freeze base

    def forward(self, x):
        result = F.linear(x, T(self.weight), bias=self.bias)
        intermediate = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        result += intermediate * self.scaling
        return result
```

**Key Design Choices:**
1. **Merge/Unmerge mechanism** for inference efficiency — weights can be merged into W for eval, unmerged for training
2. **QKV split injection** — replaces fused `qkv` linear with separate `q_proj`, `k_proj`, `v_proj` LoRA layers
3. **Fan-in/fan-out transpose support** for compatibility with different weight layouts
4. **Tiered rank**: MLP layers use `lora_rank // 4` (since mlp_ratio=4)

### Injection Strategy
```python
# dinov2/utils/train_lora_util.py
def activate_lora(model, activate_layers=("attn",), lora_rank=8, 
                  include_attn_key=False, include_attn_proj=False):
    # Recursively finds "attn" modules and calls init_lora()
    # Only Q and V by default; K and proj are optional
```

**Targets:** `attn` (q, v) and `mlp` (fc1, fc2) layers
**Default rank:** r=8 for attention, r=2 for MLP (rank//4)

### Training Loop (MAE)
```python
# mae/engine_pretrain.py
with torch.cuda.amp.autocast():
    loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
loss /= accum_iter
loss_scaler(loss, optimizer, parameters=model.parameters(), 
            update_grad=(data_iter_step + 1) % accum_iter == 0)
```

**Objective:** Standard MAE reconstruction loss on masked patches
**Optimizer:** AdamW with cosine LR scheduling
**Frozen:** All backbone weights except LoRA params + LayerNorm

### Comparison with MBPS

| Aspect | ExPLoRA | MBPS |
|--------|---------|------|
| **Backbone** | DINOv2 / MAE ViT | DINOv2 ViT-B |
| **Adapter** | Standard LoRA | LoRA / DoRA / Conv-DoRA |
| **Frozen?** | Yes, all except LoRA + LN | Yes, via `freeze_non_adapter_params()` |
| **SSL Objective** | DINO/iBOT distillation OR MAE reconstruction | Feature distillation + depth-cluster + cross-view |
| **EMA Teacher?** | Yes (DINOv2 built-in EMA) | Yes (explicit frozen teacher) |
| **Rank** | r=8 attn, r=2 MLP | r=4 default, configurable |
| **QKV handling** | Splits fused qkv into q/k/v | Wraps existing qkv linear intact |
| **Merge/unmerge** | ✅ Yes | ❌ No |
| **Spatial conv** | ❌ No | ✅ Conv-DoRA has DWConv |
| **Multi-task** | Single objective | Multi-objective (distill + cluster + consistency) |

**MBPS Advantage:** ExPLoRA only supports standard LoRA. MBPS supports DoRA and Conv-DoRA with spatial convolution paths — better for dense prediction.

**ExPLoRA Advantage:** Merge/unmerge mechanism for efficient inference. MBPS should consider adding this.

---

## 2. GDA / SLR (CVPR 2024) — MAE + Scaled Low-Rank Adapters

**Repo:** `github.com/HSG-AIML/GDA`

### Adapter Architecture: Scaled Low-Rank (SLR)
```python
# src/models.py
class ScaledLowRankAdapter(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, hidden_dim: int = 16):
        self.linear = linear
        # Freeze original
        for p in self.linear.parameters():
            p.requires_grad = False
        
        # Learnable scaling vectors
        self.in_scaler = torch.nn.Parameter(torch.ones(self.in_dim))
        self.out_scaler = torch.nn.Parameter(torch.ones(self.out_dim))
        
        # Low-rank decomposition
        self.down = torch.nn.Linear(self.in_dim, self.hidden_dim)
        self.up = torch.nn.Linear(self.hidden_dim, self.out_dim)
        
        # Init: up=0, down=normal
        self.up.weight.data.fill_(0)
        self.up.bias.data.fill_(0)
        torch.nn.init.normal_(self.down.weight.data)

    def forward(self, x):
        x_scaled = x * self.in_scaler
        x_lr = self.up(self.down(x_scaled))
        x = self.linear(x_scaled)
        x_new = x + x_lr
        x = x_new * self.out_scaler
        return x
```

**Key Design Choices:**
1. **Input/output scaling vectors** — learned channel-wise scaling (similar to FiLM/IA3)
2. **Bottleneck rank** controlled by `hidden_dim` (default 8)
3. **Residual connection** around frozen linear + low-rank branch
4. Also supports: pure LowRankAdapter (no scaling), ScalingAdapter (only scaling), IA3-style configs

### Injection Strategy
```python
# src/models.py
config = ScaledLowRankConfigTimmViT(hidden_dim=8, patch_embed=False, norm=True)
config.adapter_modules = ".*attn|.*mlp|decoder_embed|decoder_pred"
config.adapter_layers = "qkv|fc1|fc2|proj|decoder_embed|decoder_pred"
config.extra_trainable_param_names = ".*norm.*"  # Also unfreeze LayerNorm

model = add_extra_weights(model, config, ScaledLowRankAdapter, ScaledLowRankConvAdapter)
```

**Targets:** Attention (qkv, proj), MLP (fc1, fc2), decoder embed/pred, optional patch_embed
**Also unfreezes:** LayerNorm layers

### Training Loop
```python
# src/trainers/mae_adaptation.py (PyTorch Lightning)
class MaskedAutoencoding(torchgeo.trainers.base.BaseTask):
    def training_step(self, batch, batch_idx):
        x = batch["image"].squeeze()
        loss, pred, mask = self(x)  # MAE forward
        return loss  # Standard MAE reconstruction loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.95))
        scheduler = ReduceLROnPlateau(optimizer, patience=5)
```

**Objective:** MAE reconstruction only
**Optimizer:** AdamW with ReduceLROnPlateau
**Frozen:** Backbone fully frozen, adapters + LN trainable

### Comparison with MBPS

| Aspect | GDA/SLR | MBPS |
|--------|---------|------|
| **Adapter type** | SLR (low-rank + scaling vectors) | LoRA / DoRA / Conv-DoRA |
| **Scaling** | Learnable input/output channel scaling | DoRA has magnitude scaling |
| **LN unfrozen?** | Yes, via config | Optional |
| **Bottleneck** | down → up linear | A → B linear (same math) |
| **Init strategy** | up=0, down=normal | A=kaiming, B=0 (same as LoRA standard) |
| **MAE objective** | Reconstruction only | Not used |
| **Code quality** | PyTorch Lightning, modular | Pure PyTorch, custom |
| **Config system** | Regex-based module matching | Explicit function calls per layer |

**MBPS Advantage:** DoRA's magnitude decomposition is more principled than simple channel scaling. Conv-DoRA adds spatial inductive bias.

**GDA Advantage:** Very clean modular design with regex-based adapter injection. MBPS could adopt `add_extra_weights()` pattern for cleaner injection.

---

## 3. LoRA-TTT (ICML 2025) — LoRA + Entropy + MAE for Test-Time Training

**Repo:** `github.com/ykojima4020/LoRA-TTT`

### Adapter Architecture
Uses **HuggingFace PEFT** library:
```python
# factory.py
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=self._peft.r,
    target_modules=self._peft.target_modules,
    lora_alpha=(self._peft.r * self._peft.alpha_r_scale),
    lora_dropout=self._peft.dropout,
    layers_to_transform=self._peft.layers_to_transform
)
image_encoder = get_peft_model(image_encoder, config)
```

**Key Design Choices:**
1. Uses **official HuggingFace PEFT** implementation — battle-tested
2. Targets specific modules via `target_modules` list
3. `layers_to_transform` restricts LoRA to specific layer indices
4. No custom LoRA implementation — relies on PEFT

### Loss Functions (Test-Time Adaptation)
```python
# tta/tta.py
class MEMLoss:  # Marginal Entropy Minimization
    def __call__(self, model, images, text_embeddings):
        image_features = model.clip.image_encode(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        output = model.clip.logit_scale.exp() * (image_features @ text_embeddings)
        loss_output, _ = select_confident_samples(output, self._selection_p)
        loss = avg_entropy(loss_output)
        return loss

class MAELoss:
    def __call__(self, model, images, text_embeddings):
        # Select confident samples first
        _, selected_idx = select_confident_samples(output, self._selection_p)
        images = images[selected_idx]
        loss, reconstruction, mask = model.mae(images)
        return loss

class MAEMEMLossV2:
    def __call__(self, model, images, text_embeddings):
        # Combined: entropy on all, MAE on confident
        loss_output, selected_idx = select_confident_samples(output, self._selection_p)
        mem_loss = avg_entropy(loss_output)
        images = images[selected_idx]
        mae_loss, _, _ = model.mae(images)
        return (memw * mem_loss) + (maew * mae_loss)
```

**Confidence Selection:**
```python
def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx
```

**Key Design Choices:**
1. **Entropy-based confidence filtering** — only adapt on low-entropy (high-confidence) samples
2. **Multi-objective**: MEM (entropy) + MAE (reconstruction)
3. **Test-time adaptation** — adapts per-sample at inference time
4. **LoRA only on image encoder** layers 11-12 (last 2 layers)

### Training Loop
```python
# tta/tta.py — ImageEncoderTTA.update()
def update(self, images):
    self.model.train()
    for j in range(self.config.epochs):
        if self.config.reset:
            self.reset_optim()
        with torch.autocast(device_type='cuda', enabled=self.amp):
            loss = self.loss(self.model, images, self.text_embeddings)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
    return loss
```

### Comparison with MBPS

| Aspect | LoRA-TTT | MBPS |
|--------|----------|------|
| **LoRA impl** | HuggingFace PEFT | Custom (lora_layers.py) |
| **Scope** | Test-time (per-sample) | Training-time (dataset-level) |
| **Confidence filtering** | ✅ Entropy-based selection | ❌ Not used |
| **Multi-objective** | MEM + MAE | Distillation + depth-cluster + cross-view |
| **Optimizer reset** | ✅ Resets per sample | ❌ No reset |
| **AMP** | ✅ GradScaler | ❌ No AMP |
| **Teacher** | Frozen CLIP + MAE | Frozen DINOv2 + CAUSE-TR |
| **Layer restriction** | Last 2 layers only | Tiered (early=qkv, late=full) |

**MBPS Advantage:** Dataset-level training with pseudo-label generation. LoRA-TTT is test-time only — not directly comparable for stage-1 training.

**LoRA-TTT Ideas for MBPS:**
1. **Confidence filtering** — entropy-based sample selection could filter noisy pseudo-labels
2. **AMP/GradScaler** — MBPS should add mixed-precision training
3. **Optimizer state reset** — could help escape local minima during adapter warm-up

---

## 4. Uni-UVPT (NeurIPS 2023) — Visual Prompt Tuning for Segmentation UDA

**Repo:** `github.com/huawei-noah/noah-research/tree/master/uni-uvpt`

### Adapter Architecture: Prompt Adapter

**NOT LoRA-based!** Uses a **prompt generator + prompt interactor** architecture:

```python
# mmseg_custom/models/backbones/adapter_modules.py
class SpatialPriorModule(nn.Module):
    """Generates multi-scale spatial prompts from input image"""
    def __init__(self, inplanes=64, embed_dim=384):
        self.stem = nn.Sequential(...)  # Conv stem
        self.conv2 = nn.Sequential(...)  # 2x down
        self.conv3 = nn.Sequential(...)  # 4x down
        # Projects to embed_dim at multiple scales
        self.fc1 = nn.Conv2d(inplanes, embed_dim, 1)
        self.fc2 = nn.Conv2d(2*inplanes, embed_dim, 1)
        self.fc3 = nn.Conv2d(4*inplanes, embed_dim, 1)

class InteractionBlockPrompt(nn.Module):
    """Injects prompts into frozen backbone via deformable attention"""
    def __init__(self, dim, num_heads=6, n_points=4, ...):
        self.injector = Injector(dim=dim, ...)  # Deformable cross-attn
        self.extractor = Extractor(dim=dim, ...)  # FFN + deformable attn
        # Upsampling layers for next stage
        self.up_layer1 = nn.Conv2d(dim, next_dim, 3, stride=2)
```

**Key Design Choices:**
1. **Spatial Prior Module (SPM)** — lightweight conv network generating multi-scale prompts
2. **Deformable attention injector** — injects prompts into frozen backbone features
3. **Interaction blocks** at each stage — bidirectional: backbone→prompts→backbone
4. **Level embeddings** — learned embeddings for multi-scale prompt fusion

### Backbone Freezing
```python
# mmseg_custom/models/backbones/swin_prompt.py
if freeze_backbone:
    for name, params in self.named_parameters():
        if 'spm' in name or 'interactions' in name or 'level_embed' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False
```

**Frozen:** All Swin transformer parameters
**Trainable:** SPM, interaction blocks, level embeddings

### Training (MMSegmentation framework)
```python
# train.py
parser.add_argument('--freeze-backbone', action='store_true')
parser.add_argument('--prompt-lr-mult', default=5, type=float)
parser.add_argument('--feature-consistency-loss-weight', type=float, default=0.001)
parser.add_argument('--prediction-consistency-loss-weight', type=float, default=1.0)

# Optimizer: backbone frozen, prompt modules get 5x LR
optimizer.paramwise_cfg.custom_keys.level_embed.lr_mult = args.prompt_lr_mult
optimizer.paramwise_cfg.custom_keys.spm.lr_mult = args.prompt_lr_mult
optimizer.paramwise_cfg.custom_keys.interactions.lr_mult = args.prompt_lr_mult
```

**Loss Functions:**
1. **GtASelfTrainingLoss** — pseudo-label cross-entropy
2. **Feature consistency loss** (weight=0.001) — multiscale feature alignment
3. **Prediction consistency loss** (weight=1.0) — prediction stability across scales

**Pseudo-label correction:**
- Online IoU curve monitoring
- Adaptive correction based on early-learning phenomenon
- Trustable quantile filtering (default 0.66)

### Comparison with MBPS

| Aspect | Uni-UVPT | MBPS |
|--------|----------|------|
| **Adapter type** | Prompt generator + deformable injector | LoRA/DoRA in attention/MLP |
| **Backbone** | Swin / MiT | DINOv2 ViT |
| **Frozen?** | ✅ Fully frozen | ✅ Fully frozen |
| **Task** | Semantic segmentation UDA | Panoptic segmentation (stuff+thing) |
| **Pseudo-labels** | ✅ Adaptive correction | ✅ From CAUSE-TR cluster probe |
| **Multi-scale** | ✅ Feature + prediction consistency | ⚠️ Depth at single scale |
| **Loss weights** | Feature=0.001, Pred=1.0 | Configurable per loss |
| **LR multiplier** | 5× for prompt modules | Uniform LR |
| **Framework** | MMSegmentation | Pure PyTorch |

**MBPS Advantage:** Direct parameter-efficient tuning (LoRA/DoRA) vs. heavy prompt generator (SPM has many conv layers). MBPS adapters are much more parameter-efficient.

**Uni-UVPT Ideas for MBPS:**
1. **Adaptive pseudo-label correction** — monitor pseudo-label quality and correct noisy labels
2. **LR multiplier for adapters** — give adapters higher LR than other trainable params
3. **Feature consistency loss** — align features across scales/augmentations
4. **Online IoU monitoring** — track pseudo-label quality during training

---

## 5. MBPS Current Implementation Analysis

### Architecture (`mbps_pytorch/models/adapters/lora_layers.py`)

```python
class DoRALinear(nn.Module):
    def __init__(self, wrapped: nn.Linear, rank=4, alpha=4.0, dropout=0.05):
        self.register_buffer("weight", wrapped.weight.data)
        self.lora_magnitude = nn.Parameter(
            self.weight.data.norm(dim=1, keepdim=True).clone()
        )
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
    def forward(self, x):
        delta_V = self.lora_dropout(self.scaling * (lora_B @ lora_A))
        V_prime = self.weight + delta_V
        V_norm = V_prime.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_prime = self.lora_magnitude * (V_prime / V_norm.detach())
        return F.linear(x, W_prime, bias)
```

**Strengths:**
- ✅ Clean separation of concerns (layers, injection, freezing)
- ✅ Multiple variants: LoRA, DoRA, Conv-DoRA
- ✅ Conv-DoRA adds DWConv for spatial inductive bias
- ✅ Generic `wrap_linear_if_match()` helper
- ✅ `freeze_non_adapter_params()` with suffix matching

**Weaknesses vs. Paper Implementations:**
- ❌ No merge/unmerge mechanism (ExPLoRA has this)
- ❌ No confidence filtering / entropy-based selection (LoRA-TTT)
- ❌ No adaptive pseudo-label correction (Uni-UVPT)
- ❌ No AMP/GradScaler (LoRA-TTT)
- ❌ No LR multiplier for adapters (Uni-UVPT)
- ❌ No feature consistency loss across scales (Uni-UVPT)
- ❌ Single-scale depth vs. multi-scale (Uni-UVPT)

---

## 6. Actionable Recommendations for MBPS

### High Priority (Easy Wins)

1. **Add merge/unmerge for inference efficiency**
   ```python
   # Like ExPLoRA
   def merge(self):
       self.weight.data += (lora_B @ lora_A) * scaling
       self.merged = True
   
   def unmerge(self):
       self.weight.data -= (lora_B @ lora_A) * scaling
       self.merged = False
   ```

2. **Add AMP/GradScaler support**
   ```python
   # Like LoRA-TTT
   scaler = torch.GradScaler()
   with torch.cuda.amp.autocast():
       loss = ...
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **LR multiplier for adapters**
   ```python
   # Like Uni-UVPT
   adapter_params = [p for n, p in model.named_parameters() if is_adapter(n)]
   other_params = [p for n, p in model.named_parameters() if not is_adapter(n)]
   optimizer = AdamW([
       {'params': adapter_params, 'lr': lr * 5},
       {'params': other_params, 'lr': lr}
   ])
   ```

### Medium Priority (Validation Improvements)

4. **Confidence-based pseudo-label filtering**
   - Compute entropy of CAUSE-TR cluster assignments
   - Filter high-entropy (uncertain) pseudo-labels
   - Similar to LoRA-TTT's `select_confident_samples()`

5. **Feature consistency loss**
   - Add multiscale consistency between different patch resolutions
   - Similar to Uni-UVPT's feature consistency loss

6. **Adaptive pseudo-label correction**
   - Monitor pseudo-label IoU/quality curve during training
   - Correct noisy labels using early-learning phenomenon
   - Complex but high-impact (Uni-UVPT's key innovation)

### Low Priority (Research Extensions)

7. **Consider SLR-style scaling vectors** (from GDA)
   - Adds channel-wise learnable scaling to DoRA
   - More flexible than DoRA's single magnitude vector

8. **Progressive distillation** (from FORLA paper)
   - Early epochs: EMA-only teacher updates
   - Later epochs: bidirectional student→teacher transfer

---

## 7. Citation Guide for MBPS Paper

When writing about MBPS's adapter approach, cite these papers for specific design choices:

| MBPS Design Choice | Cite This Paper | For This Claim |
|-------------------|-----------------|----------------|
| Frozen backbone + adapter training | ExPLoRA (ICML 2025) | "Extending SSL pre-training with adapters on frozen ViT" |
| Adapter-only outperforms full fine-tuning | GDA/SLR (CVPR 2024) | "SLR adapters outperform full fine-tuning on MAE" |
| DoRA vs LoRA | DoRA (ICML 2024) | "Weight-decomposed adaptation closer to full fine-tuning" |
| Multi-objective SSL training | LoRA-TTT (ICML 2025) | "Entropy + reconstruction losses improve adaptation" |
| Dense prediction with frozen backbone | Uni-UVPT (NeurIPS 2023) | "Source-free UDA for segmentation with frozen backbone" |
| Spatial adapter for dense tasks | MARCO (CVPR 2026) | "Spatial adapters outperform linear LoRA for dense prediction" |
| EMA teacher-student | DINO (ICLR 2021) | "Self-distillation with EMA teacher stabilizes training" |
| Conv-DoRA novelty | **MBPS** | First to combine DoRA with spatial convolution for panoptic segmentation |

---

*End of Analysis*
