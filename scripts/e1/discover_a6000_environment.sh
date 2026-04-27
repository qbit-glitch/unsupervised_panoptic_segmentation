#!/bin/bash
# scripts/e1/discover_a6000_environment.sh
#
# Read-only A6000 discovery for E1 CUPS pseudo-label generation.
# Surfaces:
#   1. host + GPU baseline
#   2. all candidate "Data1"-like mounts (size, free space, owner)
#   3. presence of Cityscapes (left/right/sequence/disp/gtFine) under each candidate
#   4. presence of conda env `ups` and the DepthG checkpoint
#   5. existing CUPS pseudo-label directories from prior attempts
#
# Run from your local Mac:
#   ssh cvpr_ug_5@gpunode2 'bash -s' < scripts/e1/discover_a6000_environment.sh
#
# Or after cloning to the A6000:
#   bash scripts/e1/discover_a6000_environment.sh
#
# Performs NO writes, NO downloads, NO module loads, NO conda activations.

set -u
RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; CYN=$'\033[36m'; RST=$'\033[0m'

section() { echo; echo "${CYN}=== $* ===${RST}"; }
ok()      { echo "  ${GRN}OK${RST}  $*"; }
miss()    { echo "  ${RED}--${RST}  $*"; }
note()    { echo "  ${YLW}!!${RST}  $*"; }

section "1. HOST"
echo "  hostname   : $(hostname)"
echo "  whoami     : $(whoami)"
echo "  date       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  uname      : $(uname -a)"

section "2. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader \
    | sed 's/^/  /'
else
  miss "nvidia-smi not found in PATH"
fi

section "3. CANDIDATE DATA MOUNTS (size + free)"
# Largest non-system mounts, sorted by size descending
df -h --output=source,size,used,avail,pcent,target 2>/dev/null \
  | grep -v -E '(^Filesystem|tmpfs|devtmpfs|overlay|squashfs|/snap/|/boot|/proc|/sys|/dev|/run|^Mounted)' \
  | awk 'NR==1 || $2 ~ /[0-9]+G|T/' \
  | sort -k2 -h -r \
  | sed 's/^/  /'

section "4. CANDIDATE 'DATA1' DIRECTORIES BY NAME"
# Look for anything matching data1 / Data1 / data_1 / DATA1 anywhere reasonable.
for base in / /mnt /media /media/$(whoami) /data /home /home/$(whoami); do
  [ -d "$base" ] || continue
  while IFS= read -r d; do
    [ -z "$d" ] && continue
    free=$(df -h "$d" 2>/dev/null | awk 'NR==2 {print $4}')
    own=$(stat -c '%U:%G' "$d" 2>/dev/null || echo unknown)
    perms=$(stat -c '%a' "$d" 2>/dev/null || echo unknown)
    echo "  $d  (free=$free, owner=$own, perms=$perms)"
  done < <(find "$base" -maxdepth 3 -type d \( -iname 'data1' -o -iname 'data_1' -o -iname 'data-1' \) 2>/dev/null)
done | sort -u

section "5. CITYSCAPES PRESENCE UNDER EACH CANDIDATE"
# Search any directory called 'cityscapes' or 'Cityscapes' anywhere shallow.
mapfile -t CITY_ROOTS < <(find /mnt /media /data /home -maxdepth 4 -type d -iname 'cityscapes' 2>/dev/null | sort -u)

if [ ${#CITY_ROOTS[@]} -eq 0 ]; then
  miss "no directory called 'cityscapes' found under /mnt /media /data /home"
else
  for root in "${CITY_ROOTS[@]}"; do
    echo
    echo "  ${CYN}root: $root${RST}"
    for sub in leftImg8bit leftImg8bit_sequence disparity gtFine; do
      for split in train val test; do
        d="$root/$sub/$split"
        if [ -d "$d" ]; then
          n=$(find "$d" -mindepth 2 -maxdepth 2 -type f 2>/dev/null | wc -l)
          ok  "$sub/$split  ($n files)"
        else
          miss "$sub/$split"
        fi
      done
    done
    # Pseudo-label leftovers from earlier attempts
    for pl in cups_pseudo_labels cups_pseudo_labels_pipeline cups_pseudo_labels_pipeline_v2 cups_pseudo_labels_official; do
      if [ -d "$root/$pl" ]; then
        n_inst=$(find "$root/$pl" -type f \( -name '*.pt' -o -name '*.png' \) 2>/dev/null | wc -l)
        note "existing pseudo-label dir: $pl ($n_inst files)"
      fi
    done
  done
fi

section "6. CONDA ENV 'ups'"
for conda_sh in ~/miniconda3/etc/profile.d/conda.sh ~/anaconda3/etc/profile.d/conda.sh /opt/anaconda3/etc/profile.d/conda.sh /opt/miniconda3/etc/profile.d/conda.sh; do
  if [ -f "$conda_sh" ]; then
    ok "conda found: $conda_sh"
    # shellcheck disable=SC1090
    source "$conda_sh" 2>/dev/null
    if conda env list 2>/dev/null | grep -qE '^ups\s'; then
      ok "env 'ups' exists"
      ups_python=$(conda run -n ups which python 2>/dev/null)
      ups_torch=$(conda run -n ups python -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo "torch not importable")
      echo "    python : $ups_python"
      echo "    torch  : $ups_torch"
    else
      miss "env 'ups' not in conda env list"
    fi
    break
  fi
done

section "7. DEPTHG CHECKPOINT (CUPS Pass-2 semantic model)"
# Scan a few likely roots for any depthg*.ckpt
found_any=0
for root in ~ /mnt /media /data /home; do
  [ -d "$root" ] || continue
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    sz=$(du -h "$f" 2>/dev/null | cut -f1)
    sha_head=$(head -c 64 "$f" 2>/dev/null | sha256sum | cut -c1-12)
    ok "$f  ($sz, head-sha256:$sha_head)"
    found_any=1
  done < <(find "$root" -maxdepth 6 -type f -iname 'depthg*.ckpt' 2>/dev/null)
done
if [ $found_any -eq 0 ]; then
  miss "no depthg*.ckpt found under ~ /mnt /media /data /home"
fi

section "8. REPO ON A6000"
for d in ~/umesh/unsupervised_panoptic_segmentation ~/unsupervised_panoptic_segmentation ~/mbps_panoptic_segmentation ~/umesh/mbps_panoptic_segmentation; do
  if [ -d "$d/.git" ]; then
    ok "$d  (branch: $(git -C "$d" branch --show-current 2>/dev/null), HEAD: $(git -C "$d" rev-parse --short HEAD 2>/dev/null))"
  fi
done

section "9. HEADROOM CHECK"
echo "  Cityscapes Stereo+Video footprint:"
echo "    leftImg8bit_trainvaltest.zip   ~  11 GB unpacked"
echo "    leftImg8bit_sequence.zip       ~ 119 GB unpacked  (TRAIN+VAL+TEST)"
echo "    disparity_trainvaltest.zip     ~   3.5 GB unpacked"
echo "    gtFine_trainvaltest.zip        ~   0.25 GB unpacked"
echo "    cups_pseudo_labels (Pass1+2)   ~   5 GB"
echo "    --- target free space on Data1: >= 200 GB"

echo
echo "${CYN}=== DISCOVERY COMPLETE ===${RST}"
