
```bash
gcloud compute tpus tpu-vm ssh panoptic-tpu-mbps --zone=us-central2-b --command='tail -f /tmp/training.log'
```

```bash
gcloud compute tpus tpu-vm ssh panoptic-tpu-cuts3d --zone=us-central2-b --command='tail -f ~/mbps_panoptic_segmentation/logs/cuts3d_full_*.log'
```

```bash
ssh santosh@172.17.254.146 'tail -f /media/santosh/Kuldeep/panoptic_segmentation/logs/extract_coco_gpu*.log
```

```bash
gcloud compute tpus tpu-vm ssh panoptic-tpu-mbps --zone=us-central2-b --project=unsupervsed-panoptic-segment --command="tail -f /tmp/smoke_test_v2.log"
```



C++ => arduino

