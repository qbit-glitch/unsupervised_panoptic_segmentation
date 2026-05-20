# unMORE ImageNet/VoteCut Dataset Status

Date: 2026-05-18

## Downloaded

VoteCut ImageNet train pseudo-label annotations were downloaded from the unMORE README link.

- Path: `datasets/unmore_imagenet/annotations/imagenet_train_votecut_kmax_3_tuam_0.2.json`
- Size: `7027807982` bytes
- Type: COCO-style JSON
- Source ID: `10vz02vuZV1ql1QoWmMQzSQrivsIOr7Ke`

The file begins with:

```json
{"info": {"description": "ImageNet train-set: VoteCut pseudo-masks"
```

## Folder Layout

Created the local unMORE data root:

```text
datasets/unmore_imagenet/
  annotations/
    imagenet_train_votecut_kmax_3_tuam_0.2.json
  train/
  masks_top1_single_component/
  downloads/
```

## Code Wiring

`test-instance-labels/unMORE/datasets.py` no longer uses literal placeholder strings for the ImageNet/VoteCut paths.

Defaults:

- `UNMORE_IMAGENET_TRAIN` -> `datasets/unmore_imagenet/train`
- `UNMORE_VOTECUT_TOP1_MASKS` -> `datasets/unmore_imagenet/masks_top1_single_component`
- `UNMORE_VOTECUT_FULL_MASKS` -> `datasets/unmore_imagenet/masks_full`

## Still Missing

The raw ImageNet-1K train image export from Hugging Face is in progress.

- Expected path: `datasets/unmore_imagenet/train/<wnid>/*.JPEG`
- Current status: running in tmux session `unmore_imagenet_export`
- Log: `logs/export_hf_imagenet_train_for_unmore.log`
- Progress JSON: `datasets/unmore_imagenet/export_hf_imagenet_train_status.json`
- First shard verified: `4358` JPEGs written in unMORE layout.
- Hugging Face parquet paths such as `n03954731_53652_n03954731.JPEG` are converted to `n03954731/n03954731_53652.JPEG`.

The top-1 VoteCut PNG masks are also not expanded yet.

- Expected path: `datasets/unmore_imagenet/masks_top1_single_component/<wnid>/*.png`
- Current status: empty
- Reason: unMORE's provided `utils/preprocess_votecut.py` loads the full 6.5 GB JSON into memory. It should be replaced with a streaming converter before generating around ImageNet-scale PNG masks locally.
