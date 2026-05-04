# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/cascade_rcnn.py

from typing import List, Optional

import torch
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from torch import nn
from torch.autograd.function import Function

from cups.model.structures import pairwise_iou_max_scores

from .fast_rcnn import FastRCNNOutputLayers, fast_rcnn_inference
from .roi_heads import ROI_HEADS_REGISTRY, CustomStandardROIHeads
from .sam3_mask_adapter import SAM3MaskAdapter


class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


@ROI_HEADS_REGISTRY.register()
class CustomCascadeROIHeads(CustomStandardROIHeads):
    """The ROI heads that implement :paper:`Cascade R-CNN`."""

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        stage4_rare_retain_enabled: bool = False,
        stage4_rare_classes: tuple[int, ...] = (),
        stage4_rare_retain_min_rois: int = 2,
        stage4_rare_retain_relaxed_iou: float = 0.35,
        sam3_adapter: Optional[SAM3MaskAdapter] = None,
        sam3_masks_dir: str = "",
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_pooler (ROIPooler): pooler that extracts region features from given boxes
            box_heads (list[nn.Module]): box head for each cascade stage
            box_predictors (list[nn.Module]): box predictor for each cascade stage
            proposal_matchers (list[Matcher]): matcher with different IoU thresholds to
                match boxes with ground truth for each stage. The first matcher matches
                RPN proposals with ground truth, the other matchers use boxes predicted
                by the previous stage as proposals and match them with ground truth.
        """
        assert "proposal_matcher" not in kwargs, (
            "CustomCascadeROIHeads takes 'proposal_matchers=' for each stage instead " "of one 'proposal_matcher='."
        )
        # The first matcher matches RPN proposals with ground truth, done in the base class
        kwargs["proposal_matcher"] = proposal_matchers[0]
        num_stages = self.num_cascade_stages = len(box_heads)
        box_heads = nn.ModuleList(box_heads)
        box_predictors = nn.ModuleList(box_predictors)
        assert len(box_predictors) == num_stages, f"{len(box_predictors)} != {num_stages}!"
        assert len(proposal_matchers) == num_stages, f"{len(proposal_matchers)} != {num_stages}!"
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_head=box_heads,
            box_predictor=box_predictors,
            **kwargs,
        )
        self.proposal_matchers = proposal_matchers
        self.stage4_rare_retain_enabled = bool(stage4_rare_retain_enabled)
        self.stage4_rare_classes = tuple(int(c) for c in stage4_rare_classes)
        self.stage4_rare_retain_min_rois = int(stage4_rare_retain_min_rois)
        self.stage4_rare_retain_relaxed_iou = float(stage4_rare_retain_relaxed_iou)
        self.sam3_adapter = sam3_adapter
        self.sam3_masks_dir = sam3_masks_dir
        self._sam3_vit_feats: Optional[torch.Tensor] = None
        self._sam3_masks: Optional[torch.Tensor] = None
        self._sam3_padding: Optional[torch.Tensor] = None

    def set_sam3_context(
        self,
        vit_feats: torch.Tensor,
        masks: torch.Tensor,
        padding: torch.Tensor,
    ) -> None:
        """Store per-batch SAM3 context for use in _run_stage."""
        self._sam3_vit_feats = vit_feats
        self._sam3_masks = masks
        self._sam3_padding = padding

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.pop("proposal_matcher")
        ret.update(
            {
                "stage4_rare_retain_enabled": getattr(
                    cfg.MODEL.ROI_HEADS, "STAGE4_RARE_RETAIN_ENABLED", False
                ),
                "stage4_rare_classes": tuple(getattr(cfg.MODEL.ROI_HEADS, "STAGE4_RARE_CLASSES", ())),
                "stage4_rare_retain_min_rois": getattr(
                    cfg.MODEL.ROI_HEADS, "STAGE4_RARE_RETAIN_MIN_ROIS", 2
                ),
                "stage4_rare_retain_relaxed_iou": getattr(
                    cfg.MODEL.ROI_HEADS, "STAGE4_RARE_RETAIN_RELAXED_IOU", 0.35
                ),
            }
        )
        sam3_masks_dir = getattr(cfg.MODEL.ROI_BOX_HEAD, "SAM3_MASKS_DIR", "")
        ret["sam3_masks_dir"] = sam3_masks_dir
        if getattr(cfg.MODEL.ROI_BOX_HEAD, "SAM3_MASK_ADAPTER", False):
            # Infer roi_dim from the last box head output shape
            box_heads_ret = ret.get("box_heads", [])
            roi_dim = box_heads_ret[-1].output_shape.channels if box_heads_ret else 1024
            ret["sam3_adapter"] = SAM3MaskAdapter(
                vit_dim=768,
                roi_dim=roi_dim,
                adapter_dim=getattr(cfg.MODEL.ROI_BOX_HEAD, "SAM3_ADAPTER_DIM", 256),
                n_max_masks=getattr(cfg.MODEL.ROI_BOX_HEAD, "SAM3_N_MAX_MASKS", 20),
            )
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious             = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        assert len(cascade_bbox_reg_weights) == len(cascade_ious)
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CustomCascadeROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)

        box_heads, box_predictors, proposal_matchers = [], [], []
        for match_iou, bbox_reg_weights in zip(cascade_ious, cascade_bbox_reg_weights):
            box_head = build_box_head(cfg, pooled_shape)
            box_heads.append(box_head)
            box_predictors.append(
                FastRCNNOutputLayers(
                    cfg,
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights),
                )
            )
            proposal_matchers.append(Matcher([match_iou], [0, 1], allow_low_quality_matches=False))
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_heads": box_heads,
            "box_predictors": box_predictors,
            "proposal_matchers": proposal_matchers,
        }

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features, proposals, targets=None):
        """
        Args:
            features, targets: the same as in
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        """
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                # The output boxes of the previous stage are used to create the input
                # proposals of the next stage.
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                if self.training:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            predictions = self._run_stage(features, proposals, k)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        no_gt_found = False
        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                no_gt_found = False
                with storage.name_scope("stage{}".format(stage)):
                    if self.use_droploss:
                        try:
                            box_num_list = [len(x.gt_boxes) for x in proposals]
                            gt_num_list = [torch.unique(x.gt_boxes.tensor[:100], dim=0).size()[0] for x in proposals]
                        except:
                            box_num_list = [0 for x in proposals]
                            gt_num_list = [0 for x in proposals]
                            no_gt_found = True

                        if not no_gt_found:
                            # NOTE: confidence score
                            prediction_score, predictions_delta = predictions[0], predictions[1]
                            prediction_score = F.softmax(prediction_score, dim=1)[:, 0]

                            # NOTE: maximum overlapping with GT (IoU)
                            proposal_boxes = Boxes.cat([x.proposal_boxes for x in proposals])
                            predictions_bbox = predictor.box2box_transform.apply_deltas(
                                predictions_delta, proposal_boxes.tensor
                            )
                            idx_start = 0
                            iou_max_list = []
                            for idx, x in enumerate(proposals):
                                idx_end = idx_start + box_num_list[idx]
                                iou_max_list.append(
                                    pairwise_iou_max_scores(
                                        predictions_bbox[idx_start:idx_end], x.gt_boxes[: gt_num_list[idx]].tensor
                                    )
                                )
                                idx_start = idx_end
                            iou_max = torch.cat(iou_max_list, dim=0)

                            # NOTE: get the weight of each proposal
                            weights = iou_max.le(self.droploss_iou_thresh).float()
                            weights = 1 - weights.ge(1.0).float()
                            stage_losses = predictor.losses(predictions, proposals, weights=weights.detach())
                        else:
                            stage_losses = predictor.losses(predictions, proposals)
                    else:
                        stage_losses = predictor.losses(predictions, proposals)
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]

            # Average the scores across heads
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            # Use the boxes of the last head
            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            return pred_instances

    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets):
        """Match proposals with groundtruth using the matcher at the given stage. Label the proposals as foreground or
        background based on the match.

        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances

        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, proposals_per_image.proposal_boxes)
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4)))
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes

            if self.stage4_rare_retain_enabled and self.stage4_rare_classes:
                self._retain_rare_class_rois(proposals_per_image, targets_per_image)

            fg_mask = (proposals_per_image.gt_classes >= 0) & (proposals_per_image.gt_classes < self.num_classes)
            num_fg_samples.append(fg_mask.sum().item())
            num_bg_samples.append(proposals_per_image.gt_classes.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals

    @torch.no_grad()
    def _retain_rare_class_rois(self, proposals_per_image, targets_per_image) -> None:
        """Promote relaxed-IoU proposals for rare classes in later cascade stages."""
        if len(proposals_per_image) == 0 or len(targets_per_image) == 0:
            return
        device = proposals_per_image.gt_classes.device
        rare_ids = torch.tensor(self.stage4_rare_classes, device=device, dtype=proposals_per_image.gt_classes.dtype)
        rare_gt_mask = torch.isin(targets_per_image.gt_classes.to(device), rare_ids)
        if rare_gt_mask.sum() == 0:
            return

        current_rare = torch.isin(proposals_per_image.gt_classes, rare_ids).sum().item()
        needed = max(0, self.stage4_rare_retain_min_rois - int(current_rare))
        if needed == 0:
            return

        rare_targets = targets_per_image[rare_gt_mask.cpu() if rare_gt_mask.device.type == "cpu" else rare_gt_mask]
        ious = pairwise_iou(rare_targets.gt_boxes, proposals_per_image.proposal_boxes)
        if ious.numel() == 0:
            return
        max_ious, best_rare_idx = ious.max(dim=0)
        bg_mask = proposals_per_image.gt_classes == self.num_classes
        candidate_mask = bg_mask & (max_ious > self.stage4_rare_retain_relaxed_iou)
        candidate_idxs = candidate_mask.nonzero(as_tuple=False).flatten()
        if candidate_idxs.numel() == 0:
            return

        order = torch.argsort(max_ious[candidate_idxs], descending=True)
        selected = candidate_idxs[order[:needed]]
        if selected.numel() == 0:
            return
        proposals_per_image.gt_classes[selected] = rare_targets.gt_classes.to(device)[best_rare_idx[selected]]
        proposals_per_image.gt_boxes.tensor[selected] = rare_targets.gt_boxes.tensor.to(device)[best_rare_idx[selected]]

    def _run_stage(self, features, proposals, stage):
        """
        Args:
            features (list[Tensor]): #lvl input features to ROIHeads
            proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
            stage (int): the current stage

        Returns:
            Same output as `FastRCNNOutputLayers.forward()`.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # The original implementation averages the losses among heads,
        # but scale up the parameter gradients of the heads.
        # This is equivalent to adding the losses among heads,
        # but scale down the gradients on features.
        if self.training:
            box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)
        if (
            self.sam3_adapter is not None
            and stage == self.num_cascade_stages - 1
            and self._sam3_vit_feats is not None
            and self._sam3_masks is not None
            and self._sam3_padding is not None
        ):
            box_features = self.sam3_adapter(
                box_features,
                self._sam3_vit_feats,
                self._sam3_masks,
                self._sam3_padding,
                proposals,
            )
        return self.box_predictor[stage](box_features)

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)

        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            if self.training:
                # do not filter empty boxes at inference time,
                # because the scores from each stage need to be aligned and added later
                boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals
