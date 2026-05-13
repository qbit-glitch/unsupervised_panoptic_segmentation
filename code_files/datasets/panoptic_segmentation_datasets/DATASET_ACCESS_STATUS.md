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

