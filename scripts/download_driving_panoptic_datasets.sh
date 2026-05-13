#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${1:-/code_files/datasets/panoptic_segmentation_datasets}"

mkdir -p "${TARGET_ROOT}"

download() {
  local url="$1"
  local out="$2"
  if [ -s "${out}" ]; then
    echo "[skip] ${out}"
    return
  fi
  echo "[download] ${url}"
  wget -c --show-progress -O "${out}" "${url}"
}

unpack() {
  local zip_path="$1"
  local dst="$2"
  if [ ! -f "${zip_path}" ]; then
    echo "[missing] ${zip_path}"
    return 1
  fi
  echo "[unzip] ${zip_path}"
  unzip -n "${zip_path}" -d "${dst}"
}

write_status() {
  cat > "${TARGET_ROOT}/DATASET_ACCESS_STATUS.md" <<'EOF'
# Driving Panoptic Dataset Access Status

## Downloaded by this script when public URLs are reachable

- BDD-10K panoptic validation images: `bdd100k/10k_images_val.zip`
- BDD-10K panoptic train/val labels: `bdd100k/bdd100k_pan_seg_labels_trainval.zip`
- MUSES RGB frame camera package: `muses/frame_camera_trainvaltest.zip`
- MUSES panoptic annotations: `muses/gt_panoptic_trainval.zip`

## Manual or gated access required

- ACDC: use https://acdc.vision.ee.ethz.ch/ and download the RGB package plus panoptic/ground-truth annotations after accepting the dataset terms.
- Waymo Open Dataset: accept Waymo Open Dataset terms, then use the GCS bucket. The CUPS validation setup uses v1.4.0 validation:
  `gsutil -m cp -r gs://waymo_open_dataset_v_1_4_0/individual_files/validation <target>/waymo_v1_4_0/`
- IDD: create/login to an account at https://idd.insaan.iiit.ac.in/dataset/download/; the download page displays the license before access.

EOF
}

write_status

BDD_DIR="${TARGET_ROOT}/bdd100k"
mkdir -p "${BDD_DIR}"
download "https://dl.cv.ethz.ch/bdd100k/data/10k_images_val.zip" "${BDD_DIR}/10k_images_val.zip"
download "https://dl.cv.ethz.ch/bdd100k/data/bdd100k_pan_seg_labels_trainval.zip" "${BDD_DIR}/bdd100k_pan_seg_labels_trainval.zip"
unpack "${BDD_DIR}/10k_images_val.zip" "${BDD_DIR}"
unpack "${BDD_DIR}/bdd100k_pan_seg_labels_trainval.zip" "${BDD_DIR}"

MUSES_DIR="${TARGET_ROOT}/muses"
mkdir -p "${MUSES_DIR}"
download "https://muses.ethz.ch/MUSES_packages/frame_camera_trainvaltest.zip" "${MUSES_DIR}/frame_camera_trainvaltest.zip"
download "https://muses.ethz.ch/MUSES_packages/gt_panoptic_trainval.zip" "${MUSES_DIR}/gt_panoptic_trainval.zip"
unpack "${MUSES_DIR}/frame_camera_trainvaltest.zip" "${MUSES_DIR}"
unpack "${MUSES_DIR}/gt_panoptic_trainval.zip" "${MUSES_DIR}"

if [ "${DOWNLOAD_WAYMO:-0}" = "1" ]; then
  WAYMO_DIR="${TARGET_ROOT}/waymo_v1_4_0"
  mkdir -p "${WAYMO_DIR}"
  gsutil -m cp -r gs://waymo_open_dataset_v_1_4_0/individual_files/validation "${WAYMO_DIR}/"
else
  echo "[skip] Waymo validation is large and license-gated; set DOWNLOAD_WAYMO=1 after accepting Waymo terms."
fi

echo "[done] Dataset root: ${TARGET_ROOT}"
