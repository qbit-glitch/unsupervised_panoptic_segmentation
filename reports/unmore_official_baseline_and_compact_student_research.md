# Official unMORE Baseline and Compact Student Research Note

Date: 2026-05-20

## 1. Scope

This report records the official unMORE baseline evaluation we completed locally and uses it as the benchmark for a future compact student model. The report separates:

- measured official teacher metrics from local logs;
- projected student metrics, which are not yet measured;
- research-backed training routes that could let a compact student match or exceed the teacher.

The official unMORE benchmark protocol here is COCO-style class-agnostic AP/AR on COCO20K and KITTI. PQ is intentionally not reported for these benchmarks because the official unMORE evaluation path we ran is not a panoptic PQ evaluator.

## 2. Local Baseline Setup

### 2.1 Official Checkpoints

| Checkpoint | Local path | Size | Role |
|---|---:|---:|---|
| Cascade Mask R-CNN | `test-instance-labels/unMORE/checkpoints/unMORE_model.pth` | 548 MB | Official class-agnostic detector/segmenter evaluated on COCO20K and KITTI |
| Existence model | `test-instance-labels/unMORE/checkpoints/existence_model.ckpt` | 293 MB | Stage-1 object existence classifier |
| Center-boundary model | `test-instance-labels/unMORE/checkpoints/center_boundary_model.ckpt` | 3.9 GB | Stage-1 center/SDF reasoning model |

The AP benchmarks below use the official Cascade Mask R-CNN checkpoint, because this is the official unMORE evaluation route for COCO20K/KITTI.

### 2.2 Dataset Mirrors

| Dataset | Images available to evaluator | Annotation file | Notes |
|---|---:|---|---|
| COCO20K | 19,817 | `datasets/coco/annotation/coco20k_trainval_gt.json` | COCO-style instance segmentation annotations; 145,654 annotations; segmentation fields present |
| KITTI | 7,481 images used by evaluator | `datasets/kitti/annotations/trainval_cls_agnostic.json` | 35,416 annotations; segmentation fields empty, so KITTI was evaluated bbox-only |

The evaluator image directories are symlink mirrors:

- `datasets/COCO/train2014 -> /Volumes/code_files/mbps_datasets/unmore_eval/COCO/train2014`
- `datasets/kitti/JPEGImages -> /Volumes/code_files/mbps_datasets/unmore_eval/kitti/JPEGImages`

## 3. Official Baseline Results

### 3.1 COCO20K

Run log:

`logs/unmore_coco20k_cpu_eval_restart.log`

Artifacts:

- `test-instance-labels/unMORE/cad_results/eval_coco20k_official_cpu_restart/config.yaml`
- `test-instance-labels/unMORE/cad_results/eval_coco20k_official_cpu_restart/log.txt`
- `test-instance-labels/unMORE/cad_results/eval_coco20k_official_cpu_restart/inference/coco_instances_results.json`
- `test-instance-labels/unMORE/cad_results/eval_coco20k_official_cpu_restart/inference/instances_predictions.pth`

Runtime:

| Images | Device | Total inference time | Seconds / image |
|---:|---|---:|---:|
| 19,817 | CPU | 5:18:45.981 | 0.965 |

COCO20K bbox AP:

| Metric | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| bbox | 13.9188 | 25.9471 | 13.0426 | 5.3155 | 15.0404 | 29.9263 |

COCO20K bbox AR:

| Metric | AR@1 | AR@10 | AR@100 | ARs | ARm | ARl |
|---|---:|---:|---:|---:|---:|---:|
| bbox | 0.068 | 0.216 | 0.354 | 0.178 | 0.415 | 0.578 |

COCO20K mask AP:

| Metric | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| segm | 12.0052 | 23.6008 | 11.1325 | 3.1658 | 11.7335 | 26.6970 |

COCO20K mask AR:

| Metric | AR@1 | AR@10 | AR@100 | ARs | ARm | ARl |
|---|---:|---:|---:|---:|---:|---:|
| segm | 0.062 | 0.195 | 0.305 | 0.153 | 0.360 | 0.494 |

### 3.2 KITTI

Run log:

`logs/unmore_kitti_cpu_eval.log`

Artifacts:

- `test-instance-labels/unMORE/cad_results/eval_kitti_official_cpu/config.yaml`
- `test-instance-labels/unMORE/cad_results/eval_kitti_official_cpu/log.txt`
- `test-instance-labels/unMORE/cad_results/eval_kitti_official_cpu/inference/coco_instances_results.json`
- `test-instance-labels/unMORE/cad_results/eval_kitti_official_cpu/inference/instances_predictions.pth`

Runtime:

| Images | Device | Total inference time | Seconds / image |
|---:|---|---:|---:|
| 7,481 | CPU | 2:19:07.391 | 1.117 |

KITTI bbox AP:

| Metric | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| bbox | 13.5073 | 26.2485 | 12.5881 | 0.6945 | 10.2777 | 26.7976 |

KITTI bbox AR:

| Metric | AR@1 | AR@10 | AR@100 | ARs | ARm | ARl |
|---|---:|---:|---:|---:|---:|---:|
| bbox | 0.083 | 0.241 | 0.340 | 0.205 | 0.302 | 0.486 |

KITTI segmentation AP was not evaluated because all 35,416 KITTI annotations in the local JSON have empty segmentation fields.

## 4. Teacher Baseline Summary

| Dataset | Official task | Primary AP | AP50 | AP75 | Notes |
|---|---|---:|---:|---:|---|
| COCO20K | bbox | 13.9188 | 25.9471 | 13.0426 | Official detector AP |
| COCO20K | segm | 12.0052 | 23.6008 | 11.1325 | Main mask benchmark for student |
| KITTI | bbox | 13.5073 | 26.2485 | 12.5881 | Bbox-only due missing masks |

The key target for student training should be COCO20K mask AP, because it is the only completed official benchmark here that measures instance masks directly.

## 5. Projected Student Metrics

These numbers are projections, not measured results.

| Student class | Approx. model size | Expected COCO20K segm AP | Expected COCO20K bbox AP | Expected KITTI bbox AP | Interpretation |
|---|---:|---:|---:|---:|---|
| Very small | 5-15 MB | 5.5-7.5 | 7.0-9.0 | 7.5-9.5 | Too much capacity loss; likely poor AP75 |
| Practical small | 20-60 MB | 8.5-10.5 | 10.0-12.2 | 10.5-12.2 | Good compression baseline |
| Strong compact | 80-150 MB | 10.5-11.7 | 12.0-13.3 | 12.0-13.0 | Near-teacher if trained as pure distillation |
| Strong compact plus self-training | 80-150 MB | 12.3-13.8 | 14.0-15.5 | 14.5-17.0 | Plausible teacher-beating range if extra unlabeled training and filtering work |

The most likely first failure mode is AP75 degradation. A student can imitate coarse objectness and keep AP50 reasonable, while losing precise centers, boundaries, and masks. Therefore, AP75 and mask AP should be watched more carefully than AP50.

## 6. Can the Student Beat the Teacher?

### Short Answer

Yes, but not by pure distillation.

If the student is trained only to mimic unMORE predictions, the teacher acts as a ceiling. In that case the best realistic outcome is near-teacher performance with lower compute. To beat the teacher, the student needs additional information or a better optimization path:

- unlabeled data beyond the benchmark labels;
- strong noise/augmentation during student learning;
- EMA/self-training so the student becomes the next teacher;
- dense feature priors from modern self-supervised ViTs;
- instance-level and boundary-level losses, not only final mask copying;
- multi-teacher signals when available.

### Evidence From Research

1. General Instance Distillation (GID) is directly relevant to detection distillation. The CVPR 2021 paper reports a ResNet-50 RetinaNet student with GID reaching 39.1 AP on COCO, exceeding its ResNet-101 teacher at 38.1 AP. This supports the idea that instance-aware distillation can surpass the original teacher under the right objective.

2. Soft Teacher shows that teacher-student semi-supervised object detection can improve a full-COCO detector by using additional unlabeled images. The paper reports a 40.9 mAP baseline improved to 44.5 mAP, and a stronger Swin detector improved from 58.9 to 60.4 mAP; instance segmentation also improved by +1.2 AP. This is very relevant for our COCO20K/KITTI setup because unMORE already gives us pseudo labels and the datasets are unlabeled-friendly.

3. Noisy Student shows the general mechanism for teacher-beating self-training: a clean teacher generates pseudo labels, while a noisy student trains with dropout, stochastic depth, and heavy augmentation. The core lesson is that the student can generalize better than the teacher when pseudo labels are combined with stronger perturbations and iterative relabeling.

4. DINOv2 and DINOv3 support using strong self-supervised dense features as the student backbone rather than training a lightweight backbone from scratch. DINOv2 trained a 1B-parameter ViT and distilled smaller models that perform strongly on image and pixel-level benchmarks. DINOv3 goes further for dense features, using Gram anchoring and post-hoc strategies to improve resolution flexibility and dense feature quality.

5. DUNE shows a modern multi-teacher path: a single smaller encoder distilled from heterogeneous 2D/3D teachers can sometimes outperform the larger teachers on their own tasks. This suggests that a compact unMORE student can improve if it distills not only unMORE mask predictions, but also dense self-supervised features and geometry/objectness cues.

6. CrossKD is a strong detection-specific distillation mechanism. It routes student detection-head features through the teacher head, avoiding contradictory supervision between annotations and teacher predictions. On COCO, CrossKD boosts GFL ResNet-50 from 40.2 AP to 43.7 AP. This is a good template for distilling unMORE's detector head.

7. Mask Transfiner is not a distillation method, but it is relevant to our AP75/mask-boundary problem. It improves mask AP by focusing computation on error-prone boundary regions. A tiny boundary refinement branch could give a compact student better masks than the teacher's coarse outputs.

## 7. Recommended Student Direction

### 7.1 Size Target

Use the 80-150 MB range first.

Approximate parameter budget:

| Storage precision | 80 MB | 150 MB |
|---|---:|---:|
| FP32 | ~20M params | ~37.5M params |
| FP16 | ~40M params | ~75M params |

For a practical PyTorch checkpoint, an FP32 25-35M parameter model is a good first target. That gives enough room for:

- a DINOv2-S/DINOv3-S style encoder or similarly compact ViT;
- a lightweight FPN/neck;
- an anchor-free objectness head;
- a mask prototype/dynamic-mask head;
- a small boundary refinement head.

### 7.2 Architecture Proposal

Recommended student:

```text
SSL compact ViT encoder
  -> lightweight multi-scale neck
  -> objectness / box head
  -> center + boundary/SDF head
  -> prototype mask head or dynamic mask head
  -> optional tiny boundary refiner
```

This is better than trying to compress the 4 GB center-boundary checkpoint directly into a tiny clone. We should preserve unMORE's reasoning targets, but use a modern compact dense-feature encoder.

### 7.3 Training Recipe

Phase A: teacher cache generation

- Run official unMORE over ImageNet/VoteCut images, COCO20K, and KITTI.
- Store boxes, masks, scores, center fields, SDF/boundary maps, and existence scores.
- Keep low-confidence predictions, but mark them as soft/uncertain rather than hard labels.

Phase B: supervised distillation from unMORE

Losses:

- box regression loss from teacher boxes;
- class-agnostic objectness focal loss;
- mask loss on teacher masks;
- center/SDF regression from Stage-1 center-boundary model;
- existence-score distillation;
- relation or instance-level distillation inspired by GID;
- boundary-focused loss for high-frequency mask regions.

Phase C: noisy student / self-training

- Keep teacher predictions clean.
- Train student with strong augmentations: multi-scale resize, crop, color jitter, blur, CutOut/CopyPaste, stochastic depth.
- Maintain EMA student as a new teacher.
- Periodically regenerate pseudo labels.
- Use confidence thresholds separately for boxes, masks, and boundary quality.

Phase D: benchmark-only fine-tuning

- Fine-tune on COCO20K-style and KITTI-style domains without using benchmark GT labels as training labels.
- Use unlabeled images and teacher/student pseudo labels only.
- Select checkpoints by COCO20K mask AP and KITTI bbox AP on held-out evaluation.

## 8. Expected Outcome

If we do only pure teacher-student mimicry, the likely outcome is:

| Metric | Teacher | Strong compact pure KD target |
|---|---:|---:|
| COCO20K bbox AP | 13.9188 | 12.0-13.3 |
| COCO20K segm AP | 12.0052 | 10.5-11.7 |
| KITTI bbox AP | 13.5073 | 12.0-13.0 |

If we do the stronger route with SSL initialization, instance-aware distillation, boundary refinement, and noisy self-training, the plausible target becomes:

| Metric | Teacher | Teacher-beating student target |
|---|---:|---:|
| COCO20K bbox AP | 13.9188 | 14.0-15.5 |
| COCO20K segm AP | 12.0052 | 12.3-13.8 |
| KITTI bbox AP | 13.5073 | 14.5-17.0 |

The student can plausibly beat the teacher by 0.5-2.0 AP. A much larger jump, such as +4 AP mask AP on COCO20K, is possible only if the teacher pseudo labels are cleaned heavily, unlabeled data is used well, and the boundary/mask head learns information not present in the final unMORE masks.

## 9. Immediate Next Experiments

1. Build a teacher-output cache for COCO20K and KITTI from the completed official eval artifacts.
2. Train a small pilot student on 1,000 COCO20K images to verify the loss plumbing.
3. Evaluate every checkpoint with the same official COCO20K AP evaluator.
4. Add center/SDF distillation only after mask AP and bbox AP are reproducible.
5. Add noisy student training last, once pure distillation is stable.

Success thresholds:

| Stage | COCO20K segm AP | KITTI bbox AP | Decision |
|---|---:|---:|---|
| Pilot pure KD | >=7.0 | >=8.0 | Losses are working |
| Full pure KD | >=10.5 | >=12.0 | Strong compact student is viable |
| Self-training | >=12.0 | >=13.5 | Teacher-matching achieved |
| Teacher-beating | >12.0052 | >13.5073 | Student beats official unMORE baseline |

## 10. References

- General Instance Distillation for Object Detection, CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/html/Dai_General_Instance_Distillation_for_Object_Detection_CVPR_2021_paper.html
- End-to-End Semi-Supervised Object Detection with Soft Teacher, ICCV 2021: https://arxiv.org/abs/2106.09018
- Self-Training With Noisy Student Improves ImageNet Classification, CVPR 2020: https://arxiv.org/abs/1911.04252
- DINOv2: Learning Robust Visual Features without Supervision: https://arxiv.org/abs/2304.07193
- DINOv3: https://arxiv.org/abs/2508.10104
- DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers: https://arxiv.org/abs/2503.14405
- CrossKD: Cross-Head Knowledge Distillation for Object Detection: https://arxiv.org/abs/2306.11369
- Mask Transfiner for High-Quality Instance Segmentation, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.html
- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate: https://arxiv.org/abs/2504.19874

## 11. Note on TurboQuant

TurboQuant is useful for inference-time vector/KV-cache quantization, especially in large transformer-style systems. It is not a training recipe that will make a smaller vision student more accurate by itself. For our unMORE student, TurboQuant is best treated as a later compression layer after the student is already accurate. The accuracy path should be SSL initialization, object/instance distillation, boundary refinement, and noisy self-training.
