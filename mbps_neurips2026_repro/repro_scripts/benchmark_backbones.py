#!/usr/bin/env python3
"""Benchmark lightweight backbones for mobile panoptic segmentation.

Measures forward+backward throughput for each backbone + simple FPN decoder
at training resolution (512×1024).

Usage:
  python scripts/benchmark_backbones.py --device mps
  python scripts/benchmark_backbones.py --device cuda
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SimpleFPNDecoder(nn.Module):
    """Lightweight FPN decoder for panoptic segmentation."""

    def __init__(self, in_channels_list, fpn_dim=128, num_classes=19, embed_dim=16):
        super().__init__()
        # Lateral convs (align channels)
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fpn_dim, 1) for c in in_channels_list
        ])
        # Smooth convs
        self.smooths = nn.ModuleList([
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1) for _ in in_channels_list
        ])
        # Semantic head
        self.sem_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, num_classes, 1),
        )
        # Instance embedding head
        self.inst_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, embed_dim, 1),
        )

    def forward(self, features):
        # Build FPN top-down
        laterals = [l(f) for l, f in zip(self.laterals, features)]
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='bilinear', align_corners=False
            )
        # Smooth
        fpn_outs = [s(l) for s, l in zip(self.smooths, laterals)]
        # Use finest scale
        out = fpn_outs[0]
        sem = self.sem_head(out)
        inst = self.inst_head(out)
        return sem, inst


def sync_device(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def benchmark_backbone(timm_name, short_name, device, input_h=512, input_w=1024,
                       batch_size=4, num_warmup=3, num_iters=10):
    """Benchmark a single backbone + FPN decoder."""
    try:
        # Create backbone with feature extraction
        backbone = timm.create_model(timm_name, pretrained=False, features_only=True)
        channels = backbone.feature_info.channels()
        backbone_params = sum(p.numel() for p in backbone.parameters()) / 1e6

        # Create decoder
        decoder = SimpleFPNDecoder(channels, fpn_dim=128, num_classes=19, embed_dim=16)
        decoder_params = sum(p.numel() for p in decoder.parameters()) / 1e6
        total_params = backbone_params + decoder_params

        backbone = backbone.to(device).train()
        decoder = decoder.to(device).train()

        # Dummy input
        x = torch.randn(batch_size, 3, input_h, input_w, device=device)

        # Warmup
        for _ in range(num_warmup):
            feats = backbone(x)
            sem, inst = decoder(feats)
            loss = sem.sum() + inst.sum()
            loss.backward()
            sync_device(device.type)

        # Benchmark forward only
        fwd_times = []
        for _ in range(num_iters):
            x = torch.randn(batch_size, 3, input_h, input_w, device=device)
            sync_device(device.type)
            t0 = time.perf_counter()
            with torch.no_grad():
                feats = backbone(x)
                sem, inst = decoder(feats)
            sync_device(device.type)
            fwd_times.append((time.perf_counter() - t0) * 1000)

        # Benchmark forward + backward
        fwd_bwd_times = []
        for _ in range(num_iters):
            x = torch.randn(batch_size, 3, input_h, input_w, device=device, requires_grad=True)
            sync_device(device.type)
            t0 = time.perf_counter()
            feats = backbone(x)
            sem, inst = decoder(feats)
            loss = sem.sum() + inst.sum()
            loss.backward()
            sync_device(device.type)
            fwd_bwd_times.append((time.perf_counter() - t0) * 1000)

        fwd_mean = sum(fwd_times) / len(fwd_times)
        fb_mean = sum(fwd_bwd_times) / len(fwd_bwd_times)

        # Feature map sizes at finest scale
        with torch.no_grad():
            test_feats = backbone(torch.randn(1, 3, input_h, input_w, device=device))
            finest_shape = test_feats[0].shape[2:]

        # Training time estimates
        num_images = 2975  # Cityscapes train
        batches_per_epoch = num_images // batch_size
        epochs = 50
        total_batches = batches_per_epoch * epochs

        train_hours = (fb_mean / 1000) * total_batches / 3600

        result = {
            'name': short_name,
            'timm_name': timm_name,
            'backbone_params': backbone_params,
            'decoder_params': decoder_params,
            'total_params': total_params,
            'channels': channels,
            'finest_feat': f"{finest_shape[0]}×{finest_shape[1]}",
            'fwd_ms': fwd_mean,
            'fwd_bwd_ms': fb_mean,
            'imgs_per_sec': batch_size / (fb_mean / 1000),
            'train_hours': train_hours,
            'batch_size': batch_size,
        }

        del backbone, decoder
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

        return result

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "MPS" in str(e):
            # Try with smaller batch
            if batch_size > 1:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()
                return benchmark_backbone(timm_name, short_name, device, input_h, input_w,
                                         batch_size=batch_size // 2, num_warmup=num_warmup, num_iters=num_iters)
        return {'name': short_name, 'error': str(e)[:100]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--resolution', type=str, default='512x1024', help='HxW')
    args = parser.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    H, W = [int(x) for x in args.resolution.split('x')]

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Resolution: {H}×{W}, Batch: {args.batch_size}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    backbones = [
        ('repvit_m0_9.dist_450e_in1k', 'RepViT-M0.9'),
        ('mobilenetv4_conv_small.e2400_r224_in1k', 'MobileNetV4-S'),
        ('mobilenetv3_large_100.ra_in1k', 'MobileNetV3-L'),
        ('efficientnet_b0.ra_in1k', 'EfficientNet-B0'),
        ('mobilevitv2_100.cvnets_in1k', 'MobileViTv2-1.0'),
        ('ghostnet_100.in1k', 'GhostNet-1.0'),
    ]

    results = []
    for timm_name, short_name in backbones:
        print(f"Benchmarking {short_name}...", end=' ', flush=True)
        r = benchmark_backbone(timm_name, short_name, device, H, W, args.batch_size)
        if 'error' in r:
            print(f"FAILED: {r['error']}")
        else:
            print(f"fwd={r['fwd_ms']:.0f}ms  fwd+bwd={r['fwd_bwd_ms']:.0f}ms  "
                  f"train={r['train_hours']:.1f}h  ({r['total_params']:.1f}M params, B={r['batch_size']})")
        results.append(r)

    # Summary table
    print(f"\n{'='*110}")
    print(f"  Backbone Benchmark Summary — {device} @ {H}×{W}")
    print(f"{'='*110}")
    print(f"{'Model':<18} {'Params':<10} {'Channels':<22} {'Finest':<10} "
          f"{'Fwd (ms)':<10} {'F+B (ms)':<10} {'Img/s':<8} {'Train (h)':<10} {'B':<4}")
    print(f"{'-'*110}")

    for r in results:
        if 'error' in r:
            print(f"{r['name']:<18} ERROR: {r['error']}")
            continue
        ch_str = str(r['channels'])
        if len(ch_str) > 20:
            ch_str = ch_str[:20] + '..'
        print(f"{r['name']:<18} {r['total_params']:<10.1f} {ch_str:<22} {r['finest_feat']:<10} "
              f"{r['fwd_ms']:<10.0f} {r['fwd_bwd_ms']:<10.0f} {r['imgs_per_sec']:<8.1f} "
              f"{r['train_hours']:<10.1f} {r['batch_size']:<4}")

    print(f"{'='*110}")
    print(f"\nTraining estimates: 2975 Cityscapes images, {50} epochs, batch_size as shown")
    print(f"For 2× GTX 1080 Ti: divide MPS hours by ~1.5-2.5× (CUDA is faster)")
    print(f"For gradient accumulation: multiply hours by accum_steps/batch_size_ratio")


if __name__ == '__main__':
    main()
