#!/bin/bash
# Download DINOv3 ViT-B/16 weights via HuggingFace Hub (requires: hf auth login)
DEST=/home/cvpr_ug_5/.cache/torch/hub/checkpoints
mkdir -p "$DEST"
OUTFILE="$DEST/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
rm -f "$OUTFILE"
python -c "
from huggingface_hub import hf_hub_download
import shutil
f = hf_hub_download('facebook/dinov3-vitb16-pretrain-lvd1689m', 'model.safetensors')
print('Downloaded to:', f)
shutil.copy(f, '$OUTFILE')
print('Copied to torch cache')
"
echo "Result:"
ls -lh "$OUTFILE"
