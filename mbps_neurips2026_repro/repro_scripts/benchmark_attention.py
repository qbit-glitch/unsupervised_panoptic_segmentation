#!/usr/bin/env python3
"""Benchmark WindowedAttentionBlock: M4 Pro MPS vs GTX 1080 Ti (CUDA).

Measures:
  1. Forward pass throughput (ms/batch)
  2. Forward+backward pass throughput (ms/batch)
  3. Peak memory usage
  4. Throughput in images/sec

Covers both the ORIGINAL manual attention and OPTIMIZED F.scaled_dot_product_attention (SDPA).

Usage:
  python scripts/benchmark_attention.py --device mps   # On MacBook
  python scripts/benchmark_attention.py --device cuda   # On GTX 1080 Ti
  python scripts/benchmark_attention.py --device cpu    # Fallback
"""

import argparse
import time
import json
import sys
import os
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Manual attention (current implementation) ──────────────────────────────

def manual_windowed_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    bias: torch.Tensor, scale: float,
) -> torch.Tensor:
    """Original: explicit matmul + softmax + matmul."""
    attn = (q @ k.transpose(-2, -1)) * scale  # (B*nW, heads, ws*ws, ws*ws)
    attn = attn + bias.unsqueeze(0)
    attn = attn.softmax(dim=-1)
    out = attn @ v  # (B*nW, heads, ws*ws, head_dim)
    return out


# ─── SDPA attention (optimized) ─────────────────────────────────────────────

def sdpa_windowed_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    bias: torch.Tensor, scale: float,
) -> torch.Tensor:
    """Optimized: uses F.scaled_dot_product_attention with attn_mask for bias."""
    # SDPA expects attn_mask broadcastable to (B*nW, heads, ws*ws, ws*ws)
    # bias is (heads, ws*ws, ws*ws) — expand to (1, heads, ws*ws, ws*ws)
    attn_bias = bias.unsqueeze(0)
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_bias,
        scale=scale,
    )
    return out


# ─── Minimal WindowedAttentionBlock for benchmarking ────────────────────────

class BenchmarkAttentionBlock(nn.Module):
    """Stripped-down WindowedAttentionBlock for isolated attention benchmarking."""

    def __init__(self, d_model=192, window_size=8, num_heads=4, shift=False, use_sdpa=False):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.shift = shift
        self.shift_size = window_size // 2 if shift else 0
        self.scale = self.head_dim ** -0.5
        self.use_sdpa = use_sdpa

        self.norm_sem = nn.GroupNorm(1, d_model)
        self.norm_depth = nn.GroupNorm(1, d_model)
        self.cross_d2s = nn.Conv2d(d_model, d_model, 1)
        self.cross_s2d = nn.Conv2d(d_model, d_model, 1)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

        self.qkv_sem = nn.Linear(d_model, d_model * 3)
        self.qkv_depth = nn.Linear(d_model, d_model * 3)
        self.out_sem = nn.Linear(d_model, d_model)
        self.out_depth = nn.Linear(d_model, d_model)

        self.rel_pos_bias = nn.Parameter(
            torch.zeros(num_heads, (2 * window_size - 1) * (2 * window_size - 1))
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = coords.reshape(2, -1)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords[0] += window_size - 1
        rel_coords[1] += window_size - 1
        rel_coords[0] *= 2 * window_size - 1
        rel_pos_index = rel_coords.sum(0)
        self.register_buffer("rel_pos_index", rel_pos_index)

        self.ffn_sem = nn.Sequential(
            nn.GroupNorm(1, d_model),
            nn.Conv2d(d_model, d_model * 2, 1),
            nn.GELU(),
            nn.Conv2d(d_model * 2, d_model, 1),
        )
        self.ffn_depth = nn.Sequential(
            nn.GroupNorm(1, d_model),
            nn.Conv2d(d_model, d_model * 2, 1),
            nn.GELU(),
            nn.Conv2d(d_model * 2, d_model, 1),
        )

    def _window_partition(self, x, H, W):
        B, _, C = x.shape
        ws = self.window_size
        Hp = math.ceil(H / ws) * ws
        Wp = math.ceil(W / ws) * ws
        x = x.reshape(B, H, W, C)
        if Hp != H or Wp != W:
            x = F.pad(x, (0, 0, 0, Wp - W, 0, Hp - H))
        nH, nW = Hp // ws, Wp // ws
        x = x.reshape(B, nH, ws, nW, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * nH * nW, ws * ws, C)
        return x, Hp, Wp

    def _window_unpartition(self, x, B, Hp, Wp, H, W):
        ws = self.window_size
        C = x.shape[-1]
        nH, nW = Hp // ws, Wp // ws
        x = x.reshape(B, nH, nW, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        if Hp != H or Wp != W:
            x = x[:, :H, :W, :].contiguous()
        return x.reshape(B, H * W, C)

    def _windowed_attention(self, x, qkv_proj, out_proj, H, W):
        B = x.shape[0]
        ws = self.window_size

        if self.shift_size > 0:
            x_2d = x.reshape(B, H, W, self.d_model)
            x_2d = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x = x_2d.reshape(B, H * W, self.d_model)

        x_win, Hp, Wp = self._window_partition(x, H, W)
        nW_total = x_win.shape[0]

        qkv = qkv_proj(x_win).reshape(nW_total, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        bias = self.rel_pos_bias[:, self.rel_pos_index.view(-1)].reshape(
            self.num_heads, ws * ws, ws * ws
        )

        if self.use_sdpa:
            out = sdpa_windowed_attention(q, k, v, bias, self.scale)
        else:
            out = manual_windowed_attention(q, k, v, bias, self.scale)

        out = out.transpose(1, 2).reshape(nW_total, ws * ws, self.d_model)
        out = out_proj(out)
        out = self._window_unpartition(out, B, Hp, Wp, H, W)

        if self.shift_size > 0:
            out_2d = out.reshape(B, H, W, self.d_model)
            out_2d = torch.roll(out_2d, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            out = out_2d.reshape(B, H * W, self.d_model)

        return out

    def forward(self, sem, depth_feat):
        B, D, H, W = sem.shape
        sem_n = self.norm_sem(sem)
        depth_n = self.norm_depth(depth_feat)
        sem_input = sem_n + self.alpha * torch.sigmoid(self.cross_d2s(depth_n))
        depth_input = depth_n + self.beta * torch.sigmoid(self.cross_s2d(sem_n))
        sem_flat = sem_input.permute(0, 2, 3, 1).reshape(B, H * W, D)
        depth_flat = depth_input.permute(0, 2, 3, 1).reshape(B, H * W, D)
        sem_attn = self._windowed_attention(sem_flat, self.qkv_sem, self.out_sem, H, W)
        depth_attn = self._windowed_attention(depth_flat, self.qkv_depth, self.out_depth, H, W)
        sem_attn = sem_attn.reshape(B, H, W, D).permute(0, 3, 1, 2)
        depth_attn = depth_attn.reshape(B, H, W, D).permute(0, 3, 1, 2)
        sem_out = sem + sem_attn
        depth_out = depth_feat + depth_attn
        sem_out = sem_out + self.ffn_sem(sem_out)
        depth_out = depth_out + self.ffn_depth(depth_out)
        return sem_out, depth_out


def sync_device(device_type: str):
    """Synchronize device for accurate timing."""
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def measure_memory(device_type: str) -> float:
    """Return peak memory in MB."""
    if device_type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    elif device_type == "mps":
        # MPS memory tracking (PyTorch >= 2.1)
        try:
            return torch.mps.current_allocated_memory() / 1e6
        except AttributeError:
            return 0.0
    return 0.0


def reset_memory(device_type: str):
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()


def benchmark_config(
    device: torch.device,
    batch_size: int,
    H: int, W: int,
    d_model: int,
    use_sdpa: bool,
    num_warmup: int = 5,
    num_iters: int = 20,
    measure_backward: bool = True,
) -> dict:
    """Benchmark a single configuration. Returns timing and memory stats."""
    device_type = device.type
    label = "SDPA" if use_sdpa else "Manual"

    # Create 2 blocks (one shifted, one not) to match real decoder stage
    block0 = BenchmarkAttentionBlock(d_model=d_model, window_size=8, num_heads=4, shift=False, use_sdpa=use_sdpa).to(device)
    block1 = BenchmarkAttentionBlock(d_model=d_model, window_size=8, num_heads=4, shift=True, use_sdpa=use_sdpa).to(device)
    block0.train()
    block1.train()

    # Dummy input
    sem = torch.randn(batch_size, d_model, H, W, device=device, requires_grad=True)
    depth = torch.randn(batch_size, d_model, H, W, device=device, requires_grad=True)

    # Warmup
    for _ in range(num_warmup):
        s, d = block0(sem, depth)
        s, d = block1(s, d)
        if measure_backward:
            loss = s.sum() + d.sum()
            loss.backward()
        sync_device(device_type)

    reset_memory(device_type)

    # Forward-only timing
    fwd_times = []
    for _ in range(num_iters):
        sem_in = torch.randn(batch_size, d_model, H, W, device=device)
        depth_in = torch.randn(batch_size, d_model, H, W, device=device)
        sync_device(device_type)

        t0 = time.perf_counter()
        with torch.no_grad():
            s, d = block0(sem_in, depth_in)
            s, d = block1(s, d)
        sync_device(device_type)
        fwd_times.append((time.perf_counter() - t0) * 1000)

    fwd_mem = measure_memory(device_type)
    reset_memory(device_type)

    # Forward+backward timing
    fwd_bwd_times = []
    if measure_backward:
        for _ in range(num_iters):
            sem_in = torch.randn(batch_size, d_model, H, W, device=device, requires_grad=True)
            depth_in = torch.randn(batch_size, d_model, H, W, device=device, requires_grad=True)
            sync_device(device_type)

            t0 = time.perf_counter()
            s, d = block0(sem_in, depth_in)
            s, d = block1(s, d)
            loss = s.sum() + d.sum()
            loss.backward()
            sync_device(device_type)
            fwd_bwd_times.append((time.perf_counter() - t0) * 1000)

    bwd_mem = measure_memory(device_type)

    # Stats
    fwd_mean = sum(fwd_times) / len(fwd_times)
    fwd_std = (sum((t - fwd_mean) ** 2 for t in fwd_times) / len(fwd_times)) ** 0.5

    result = {
        "label": label,
        "device": str(device),
        "batch_size": batch_size,
        "resolution": f"{H}x{W}",
        "d_model": d_model,
        "fwd_ms_mean": round(fwd_mean, 2),
        "fwd_ms_std": round(fwd_std, 2),
        "fwd_imgs_per_sec": round(batch_size / (fwd_mean / 1000), 1),
        "fwd_peak_mem_MB": round(fwd_mem, 1),
    }

    if fwd_bwd_times:
        fb_mean = sum(fwd_bwd_times) / len(fwd_bwd_times)
        fb_std = (sum((t - fb_mean) ** 2 for t in fwd_bwd_times) / len(fwd_bwd_times)) ** 0.5
        result["fwd_bwd_ms_mean"] = round(fb_mean, 2)
        result["fwd_bwd_ms_std"] = round(fb_std, 2)
        result["fwd_bwd_imgs_per_sec"] = round(batch_size / (fb_mean / 1000), 1)
        result["fwd_bwd_peak_mem_MB"] = round(bwd_mem, 1)

    return result


def print_table(results: list[dict], title: str):
    """Print results as a formatted table."""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"{'Config':<22} {'Fwd (ms)':<14} {'Fwd+Bwd (ms)':<16} {'Img/s (train)':<14} {'Mem (MB)':<12} {'Speedup':<8}")
    print(f"{'-'*90}")

    # Group by resolution for speedup calculation
    by_res = {}
    for r in results:
        key = r["resolution"]
        if key not in by_res:
            by_res[key] = []
        by_res[key].append(r)

    for r in results:
        res = r["resolution"]
        config = f"{r['label']} {res} B={r['batch_size']}"
        fwd = f"{r['fwd_ms_mean']:.1f} ± {r['fwd_ms_std']:.1f}"
        fb = f"{r.get('fwd_bwd_ms_mean', 0):.1f} ± {r.get('fwd_bwd_ms_std', 0):.1f}"
        ips = f"{r.get('fwd_bwd_imgs_per_sec', r['fwd_imgs_per_sec']):.1f}"
        mem = f"{r.get('fwd_bwd_peak_mem_MB', r['fwd_peak_mem_MB']):.0f}"

        # Speedup: SDPA vs Manual at same resolution
        speedup = ""
        group = by_res.get(res, [])
        if len(group) == 2 and r["label"] == "SDPA":
            manual = [x for x in group if x["label"] == "Manual"][0]
            fb_key = "fwd_bwd_ms_mean"
            if fb_key in r and fb_key in manual:
                sp = manual[fb_key] / r[fb_key]
                speedup = f"{sp:.2f}x"

        print(f"{config:<22} {fwd:<14} {fb:<16} {ips:<14} {mem:<12} {speedup:<8}")

    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark WindowedAttentionBlock")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "mps", "cuda", "cpu"],
                        help="Device to benchmark on")
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_iters", type=int, default=30)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results as JSON to this path")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    elif device.type == "mps":
        print(f"Apple Silicon MPS backend")

    # Check SDPA availability
    has_sdpa = hasattr(F, "scaled_dot_product_attention")
    print(f"F.scaled_dot_product_attention available: {has_sdpa}")
    if has_sdpa and device.type == "cuda":
        print(f"  Flash Attention: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"  Memory-efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
        print(f"  Math fallback: {torch.backends.cuda.math_sdp_enabled()}")

    # Benchmark configurations matching actual training
    configs = [
        # (batch_size, H, W, d_model) — matches Phase 2/3 decoder stages
        (4, 64, 128, 192),    # 2-stage decoder, stage 1 (batch=4, P2-B config)
        (4, 128, 256, 192),   # 2-stage decoder, stage 2 (final output)
        (2, 128, 256, 192),   # 3-stage decoder, stage 2 (batch=2, P2-C/P3-B config)
        (2, 256, 512, 192),   # 3-stage decoder, stage 3 (256×512 output)
        (1, 256, 512, 192),   # 4-stage decoder, stage 3 (batch=1, P3-E config)
    ]
    # NOTE: 512×1024 omitted — exceeds 11GB GTX 1080 Ti VRAM and is very slow on MPS

    all_results = []

    for batch_size, H, W, d_model in configs:
        print(f"\n--- Benchmarking: B={batch_size}, {H}×{W}, d={d_model} ---")

        # Manual attention
        try:
            r_manual = benchmark_config(
                device, batch_size, H, W, d_model,
                use_sdpa=False,
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
            )
            all_results.append(r_manual)
            print(f"  Manual: fwd={r_manual['fwd_ms_mean']:.1f}ms, fwd+bwd={r_manual.get('fwd_bwd_ms_mean', 0):.1f}ms")
        except RuntimeError as e:
            print(f"  Manual: FAILED ({e})")
            all_results.append({
                "label": "Manual", "device": str(device), "batch_size": batch_size,
                "resolution": f"{H}x{W}", "d_model": d_model, "error": str(e),
                "fwd_ms_mean": 0, "fwd_ms_std": 0, "fwd_imgs_per_sec": 0, "fwd_peak_mem_MB": 0,
            })

        reset_memory(device.type)

        # SDPA attention
        if has_sdpa:
            try:
                r_sdpa = benchmark_config(
                    device, batch_size, H, W, d_model,
                    use_sdpa=True,
                    num_warmup=args.num_warmup,
                    num_iters=args.num_iters,
                )
                all_results.append(r_sdpa)
                print(f"  SDPA:   fwd={r_sdpa['fwd_ms_mean']:.1f}ms, fwd+bwd={r_sdpa.get('fwd_bwd_ms_mean', 0):.1f}ms")
            except RuntimeError as e:
                print(f"  SDPA: FAILED ({e})")
                all_results.append({
                    "label": "SDPA", "device": str(device), "batch_size": batch_size,
                    "resolution": f"{H}x{W}", "d_model": d_model, "error": str(e),
                    "fwd_ms_mean": 0, "fwd_ms_std": 0, "fwd_imgs_per_sec": 0, "fwd_peak_mem_MB": 0,
                })

        reset_memory(device.type)

    # Print summary table
    print_table(all_results, f"WindowedAttentionBlock Benchmark — {device}")

    # Save JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"device": str(device), "pytorch": torch.__version__, "results": all_results}, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        default_path = f"reports/benchmark_attention_{device.type}.json"
        os.makedirs("reports", exist_ok=True)
        with open(default_path, "w") as f:
            json.dump({"device": str(device), "pytorch": torch.__version__, "results": all_results}, f, indent=2)
        print(f"\nResults saved to {default_path}")


if __name__ == "__main__":
    main()
