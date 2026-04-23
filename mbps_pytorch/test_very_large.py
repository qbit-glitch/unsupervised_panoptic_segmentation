import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class LoRALinear(nn.Module):
    def __init__(self, wrapped: nn.Linear, rank: int = 4, alpha: float = 4.0, dropout: float = 0.05):
        super().__init__()
        self.in_features = wrapped.in_features
        self.out_features = wrapped.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.weight = nn.Parameter(wrapped.weight.data.clone(), requires_grad=False)
        if wrapped.bias is not None:
            self.bias = nn.Parameter(wrapped.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = self.lora_dropout(
            F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        )
        return base_out + lora_out

class DoRALinear(nn.Module):
    def __init__(self, wrapped: nn.Linear, rank: int = 4, alpha: float = 4.0, dropout: float = 0.05):
        super().__init__()
        self.in_features = wrapped.in_features
        self.out_features = wrapped.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.weight = nn.Parameter(wrapped.weight.data.clone(), requires_grad=False)
        if wrapped.bias is not None:
            self.bias = nn.Parameter(wrapped.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.lora_magnitude = nn.Parameter(self.weight.data.norm(dim=1, keepdim=True).clone())
        self.register_buffer("_m_init_norm", self.lora_magnitude.data.norm().clone())
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_V = self.lora_dropout(self.scaling * (self.lora_B @ self.lora_A))
        V_prime = self.weight + delta_V
        V_norm = V_prime.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_prime = self.lora_magnitude * (V_prime / V_norm.detach())
        return F.linear(x, W_prime, self.bias)

class ConvDoRALinear(DoRALinear):
    def __init__(self, wrapped: nn.Linear, rank: int = 4, alpha: float = 4.0, dropout: float = 0.05):
        super().__init__(wrapped, rank=rank, alpha=alpha, dropout=dropout)
        self.dwconv = nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank, bias=False)
        nn.init.zeros_(self.dwconv.weight)
        self.conv_gate = nn.Parameter(torch.zeros(1))
        self._spatial_dims: Optional[Tuple[int, int]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_V = self.lora_dropout(self.scaling * (self.lora_B @ self.lora_A))
        V_prime = self.weight + delta_V
        V_norm = V_prime.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_prime = self.lora_magnitude * (V_prime / V_norm.detach())
        dora_out = F.linear(x, W_prime, self.bias)
        h_p, w_p = self._spatial_dims or (None, None)
        if h_p is not None and w_p is not None:
            B_size = x.shape[0]
            n_patches = h_p * w_p
            seq_len = x.shape[1]
            n_special = seq_len - n_patches
            z = F.linear(x, self.lora_A)
            z_patch = z[:, n_special:, :]
            z_2d = z_patch.transpose(1, 2).reshape(B_size, self.rank, h_p, w_p)
            z_conv = self.dwconv(z_2d)
            z_flat = z_conv.reshape(B_size, self.rank, -1).transpose(1, 2)
            if n_special > 0:
                zeros = torch.zeros(B_size, n_special, self.rank, device=z.device, dtype=z.dtype)
                z_flat = torch.cat([zeros, z_flat], dim=1)
            conv_out = F.linear(z_flat, self.lora_B) * self.scaling
            dora_out = dora_out + self.conv_gate * conv_out
        return dora_out

class LoRAConv2d(nn.Module):
    def __init__(self, wrapped: nn.Conv2d, rank: int = 4, alpha: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.in_channels = wrapped.in_channels
        self.out_channels = wrapped.out_channels
        self.kernel_size = wrapped.kernel_size
        self.stride = wrapped.stride
        self.padding = wrapped.padding
        self.rank = rank
        self.scaling = alpha / rank
        self.weight = nn.Parameter(wrapped.weight.data.clone(), requires_grad=False)
        if wrapped.bias is not None:
            self.bias = nn.Parameter(wrapped.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.lora_A = nn.Conv2d(self.in_channels, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(rank, self.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        lora_out = self.lora_dropout(self.lora_B(self.lora_A(x)) * self.scaling)
        return base_out + lora_out

ADAPTER_CLASSES = {
    "lora": LoRALinear,
    "dora": DoRALinear,
    "conv_dora": ConvDoRALinear,
}

def wrap_linear_if_match(parent, attr_name, adapter_cls, rank, alpha, dropout):
    original = getattr(parent, attr_name, None)
    if original is None or not isinstance(original, nn.Linear):
        return None
    adapter = adapter_cls(wrapped=original, rank=rank, alpha=alpha, dropout=dropout)
    setattr(parent, attr_name, adapter)
    return adapter.trainable_count()

def wrap_conv2d_if_match(parent, attr_name, rank, alpha, dropout=0.0):
    original = getattr(parent, attr_name, None)
    if original is None or not isinstance(original, nn.Conv2d):
        return None
    adapter = LoRAConv2d(wrapped=original, rank=rank, alpha=alpha, dropout=dropout)
    setattr(parent, attr_name, adapter)
    return adapter.trainable_count()

def freeze_non_adapter_params(model):
    for name, param in model.named_parameters():
        if any(k in name for k in ("lora_", "dwconv", "conv_gate")):
            param.requires_grad = True
        else:
            param.requires_grad = False

def count_adapter_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_params(model):
    return sum(p.numel() for p in model.parameters())
