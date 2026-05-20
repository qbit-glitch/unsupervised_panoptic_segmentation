# unMORE Official Stage 1 Status

Date: 2026-05-18

## DPT-Large Backbone

Downloaded the DPT-Large / ViT-L/16 384 backbone used by the unMORE DPT wrapper:

- Path: `weights/unmore/dpt_large_cache/vit_large_patch16_384_augreg_in21k_ft_in1k.npz`
- Size: 1.1 GB
- SHA256: `8700cf38ba82082d95347ad16ea3294d588b359077a7c46da9da6618165c3288`
- Source URL: `https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz`

Verification:

- `timm.create_model("vit_large_patch16_384", pretrained=True, pretrained_cfg_overlay={"file": path})` loads successfully.
- unMORE `DPT(head=None, backbone="vitl16_384")` loads successfully with `UNMORE_DPT_LARGE_WEIGHTS=<path>`.
- unMORE `ObjectnessNet(backbone_type="dpt_large")` initializes with a 2-channel center head and 1-channel boundary/SDF head.

## Local Code Hook

`test-instance-labels/unMORE/models/dpt/vit.py` now honors `UNMORE_DPT_LARGE_WEIGHTS` for the DPT-Large backbone. This avoids a network call in current `timm`, which otherwise tries Hugging Face first for `pretrained=True`.

## Stage 1 Checkpoint Blocker

The official Stage 1 entrypoint was run with:

```bash
env MPLCONFIGDIR=/private/tmp/mbps_matplotlib \
  UNMORE_DPT_LARGE_WEIGHTS=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/weights/unmore/dpt_large_cache/vit_large_patch16_384_augreg_in21k_ft_in1k.npz \
  PYTHONPATH=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/test-instance-labels/unMORE \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/.venv/bin/python object_reasoning.py \
  --sdf_activation tanh --use_bg_sdf \
  --objectness_resume /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/weights/unmore/center_boundary_model.pth \
  --binary_classifier_resume /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/weights/unmore/existence_model.pth \
  --start_idx 0 --end_idx 1 --dataset COCO --dataset_split test --analyze_cc
```

Result:

- The script reaches `Restoring objectness_model checkpoint`.
- It fails at `torch.load(center_boundary_model.pth)`.
- `file weights/unmore/center_boundary_model.pth weights/unmore/existence_model.pth` reports both as HTML documents.
- The first bytes of both files start with `<!DOCTYPE html ...>`, so they are Microsoft/OneDrive HTML responses, not model checkpoints.

## Metrics Status

No official unMORE Stage 1 instance metrics were produced. Running PQ/AP with only the generic DPT-Large backbone would be invalid because the unMORE center/boundary heads and existence classifier remain untrained unless the official `center_boundary_model.pth` and `existence_model.pth` checkpoints are recovered.
