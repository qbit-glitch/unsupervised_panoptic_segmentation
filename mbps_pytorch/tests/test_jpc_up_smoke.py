"""Smoke tests for the JPC-Up panoptic pseudo-label module."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from mbps_pytorch.build_jpc_up_cache import load_bank_prior_maps, optional_proposal_masks
from mbps_pytorch.export_jpc_up_pseudolabels import output_to_pseudolabels
from mbps_pytorch.gate_instance_bank_by_semantics import gate_one, load_anchor_bank
from mbps_pytorch.generate_rare_thing_anchor_bank import (
    candidate_features,
    class_is_plausible,
    load_base_anchors,
    score_rare_classes,
    select_rare_anchors,
)
from mbps_pytorch.jpc_up_data import JPCUpCacheDataset, jpc_up_collate
from mbps_pytorch.losses.jpc_up_loss import (
    JPCUpLossConfig,
    compute_jpc_up_losses,
    mask_background_suppression_loss,
    mask_coverage_loss,
    overlap_loss,
    proposal_mask_loss,
    query_score_loss,
)
from mbps_pytorch.models.panoptic import LatentConsensusBuilder, LatentConsensusConfig
from mbps_pytorch.models.panoptic.jpc_up import JPCUpConfig, JointPanopticCouplerUp
from mbps_pytorch.train_jpc_up import _class_keep_probs_from_counts, _proposal_class_targets


def _tiny_config() -> JPCUpConfig:
    return JPCUpConfig(
        appearance_dim=16,
        semantic_dim=10,
        fusion_dim=32,
        num_prototypes=7,
        num_queries=5,
        num_query_classes=20,
        query_dim=32,
        instance_embed_dim=8,
        low_hw=(4, 8),
        mid_hw=(8, 16),
        high_hw=(16, 32),
        num_coupling_blocks=1,
        mid_num_coupling_blocks=1,
        num_heads=4,
        up_guidance_ch=24,
    )


def test_jpc_up_forward_shapes_cpu():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = JointPanopticCouplerUp(cfg)
    assert model.low_input_proj.net[0].in_channels == cfg.fusion_dim + cfg.semantic_dim + 16 + 2 + 1 + 2
    assert model.code_up1.guidance[0].net[0].in_channels == 24
    assert len(model.mid_couplers) == cfg.mid_num_coupling_blocks
    appearance = torch.randn(2, cfg.appearance_dim, *cfg.low_hw)
    semantic_codes = torch.randn(2, cfg.semantic_dim, *cfg.low_hw)
    depth = torch.rand(2, 1, 32, 64)
    rgb = torch.rand(2, 3, 32, 64)

    out = model(appearance, semantic_codes, depth, rgb)

    assert out.refined_codes.shape == (2, cfg.semantic_dim, *cfg.high_hw)
    assert out.semantic_logits.shape == (2, cfg.num_prototypes, *cfg.high_hw)
    assert out.objectness_logits.shape == (2, 1, *cfg.high_hw)
    assert out.center_logits.shape == (2, 1, *cfg.high_hw)
    assert out.boundary_logits.shape == (2, 1, *cfg.high_hw)
    assert out.instance_embeddings.shape == (2, cfg.instance_embed_dim, *cfg.high_hw)
    assert out.mask_logits.shape == (2, cfg.num_queries, *cfg.high_hw)
    assert out.query_scores.shape == (2, cfg.num_queries)
    assert out.query_class_logits.shape == (2, cfg.num_queries, cfg.num_query_classes)
    assert out.query_embeddings.shape == (2, cfg.num_queries, cfg.instance_embed_dim)
    assert out.mid_codes.shape == (2, cfg.semantic_dim, *cfg.mid_hw)
    assert out.mid_features.shape == (2, cfg.fusion_dim, *cfg.mid_hw)
    assert set(out.depth_maps) == {"low", "mid", "high"}


def test_direct_class_balancing_drops_overrepresented_targets():
    masks = torch.zeros(1, 4, 4, 4)
    masks[:, 0, :2, :2] = 1.0
    masks[:, 1, 2:, :2] = 1.0
    masks[:, 2, :2, 2:] = 1.0
    masks[:, 3, 2:, 2:] = 1.0
    semantic = torch.full((1, 4, 4), 255.0)
    semantic[:, :2, :2] = 13
    semantic[:, 2:, :2] = 13
    semantic[:, :2, 2:] = 11
    semantic[:, 2:, 2:] = 12
    scores = torch.ones(1, 4)
    keep_probs = torch.ones(19)
    keep_probs[13] = 0.0

    class_ids, balanced_scores = _proposal_class_targets(
        masks,
        scores,
        semantic,
        None,
        thing_only=True,
        semantic_target_is_trainid=True,
        class_keep_probs=keep_probs,
        max_per_class=1,
    )

    assert class_ids is not None
    assert balanced_scores is not None
    assert balanced_scores[0, :2].sum() == 0.0
    assert class_ids[0, :2].tolist() == [255, 255]
    assert class_ids[0, 2:].tolist() == [11, 12]
    assert balanced_scores[0, 2:].tolist() == [1.0, 1.0]


def test_direct_class_targets_can_use_ap_gated_bank_class_ids():
    masks = torch.zeros(1, 4, 4, 4)
    masks[:, 0, :2, :2] = 1.0
    masks[:, 1, 2:, :2] = 1.0
    masks[:, 2, :2, 2:] = 1.0
    masks[:, 3, 2:, 2:] = 1.0
    scores = torch.ones(1, 4)
    preset_classes = torch.tensor([[11, 13, 18, 0]])
    keep_probs = torch.ones(19)
    keep_probs[13] = 0.0

    class_ids, balanced_scores = _proposal_class_targets(
        masks,
        scores,
        None,
        None,
        thing_only=True,
        semantic_target_is_trainid=False,
        class_keep_probs=keep_probs,
        max_per_class=1,
        preset_class_ids=preset_classes,
    )

    assert class_ids is not None
    assert balanced_scores is not None
    assert class_ids[0].tolist() == [11, 255, 18, 255]
    assert balanced_scores[0].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_jpc_up_dataset_loads_external_direct_target_bank(tmp_path):
    city = "aachen"
    stem = "aachen_000000_000019_leftImg8bit"
    cache_dir = tmp_path / "cache" / "train" / city
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / f"{stem}.npz"
    np.savez_compressed(
        cache_path,
        appearance_features=np.zeros((16, 4, 8), dtype=np.float32),
        semantic_codes=np.zeros((10, 4, 8), dtype=np.float32),
        depth=np.zeros((16, 32), dtype=np.float32),
        stem=np.asarray(stem),
    )
    target_masks = np.zeros((2, 16, 32), dtype=bool)
    target_masks[0, 1:5, 2:7] = True
    target_masks[1, 8:13, 20:29] = True
    target_dir = tmp_path / "targets" / "train" / city
    target_dir.mkdir(parents=True)
    np.savez_compressed(
        target_dir / f"{stem}.npz",
        masks=target_masks,
        scores=np.array([0.9, 0.7], dtype=np.float32),
        class_ids=np.array([11, 18], dtype=np.int32),
        num_valid=2,
    )

    dataset = JPCUpCacheDataset(
        [cache_path],
        low_hw=(4, 8),
        direct_target_dir=tmp_path / "targets",
        direct_target_split="train",
        direct_target_required=True,
    )
    sample = dataset[0]
    batch = jpc_up_collate([sample])

    assert sample["city"] == city
    assert torch.equal(sample["direct_proposal_class_ids"], torch.tensor([11, 18]))
    assert batch["direct_proposal_masks"].shape == (1, 2, 16, 32)
    assert torch.allclose(batch["direct_proposal_scores"], torch.tensor([[0.9, 0.7]]))
    assert batch["direct_proposal_class_ids"].tolist() == [[11, 18]]


def test_direct_class_targets_vote_after_mapping_clusters_to_trainids():
    masks = torch.ones(1, 1, 4, 4)
    semantic = torch.tensor(
        [
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [2, 255, 255, 255],
            ]
        ],
        dtype=torch.long,
    )
    cluster_lut = torch.full((256,), 255, dtype=torch.long)
    cluster_lut[0] = 13
    cluster_lut[1] = 13
    cluster_lut[2] = 0

    class_ids, balanced_scores = _proposal_class_targets(
        masks,
        torch.ones(1, 1),
        semantic,
        cluster_lut,
        thing_only=True,
        semantic_target_is_trainid=False,
    )

    assert class_ids is not None
    assert balanced_scores is not None
    assert int(class_ids[0, 0]) == 13
    assert float(balanced_scores[0, 0]) == 1.0


def test_direct_class_targets_can_use_thing_fraction_vote():
    masks = torch.ones(1, 1, 4, 4)
    semantic = torch.tensor(
        [
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [11, 11, 11, 12],
                [12, 255, 255, 255],
            ]
        ],
        dtype=torch.long,
    )

    class_ids, balanced_scores = _proposal_class_targets(
        masks,
        torch.ones(1, 1),
        semantic,
        None,
        thing_only=True,
        semantic_target_is_trainid=True,
        vote_mode="thing_fraction",
    )

    assert class_ids is not None
    assert balanced_scores is not None
    assert int(class_ids[0, 0]) == 11
    assert float(balanced_scores[0, 0]) == 1.0


def test_direct_class_keep_probs_second_max_balances_dominant_class():
    counts = torch.zeros(19, dtype=torch.long)
    counts[11] = 45
    counts[13] = 637
    counts[18] = 14

    keep_probs = _class_keep_probs_from_counts(counts, mode="second_max", min_keep_prob=0.05)

    assert keep_probs is not None
    assert float(keep_probs[11]) == 1.0
    assert float(keep_probs[18]) == 1.0
    assert 0.06 < float(keep_probs[13]) < 0.08


def test_jpc_up_proposal_conditioning_anchors_masks():
    torch.manual_seed(10)
    cfg = _tiny_config()
    model = JointPanopticCouplerUp(cfg)
    appearance = torch.randn(1, cfg.appearance_dim, *cfg.low_hw)
    semantic_codes = torch.randn(1, cfg.semantic_dim, *cfg.low_hw)
    depth = torch.rand(1, 1, 32, 64)
    rgb = torch.rand(1, 3, 32, 64)
    proposal_masks = torch.zeros(1, 2, *cfg.high_hw)
    proposal_masks[:, 0, 3:12, 5:18] = 1.0
    proposal_masks[:, 1, 1:5, 20:28] = 1.0
    proposal_scores = torch.tensor([[0.8, 0.2]])

    out = model(appearance, semantic_codes, depth, rgb, proposal_masks, proposal_scores)

    assert out.proposal_anchor_masks is not None
    assert out.proposal_anchor_logits is not None
    assert out.proposal_anchor_valid is not None
    assert out.proposal_anchor_indices is not None
    assert bool(out.proposal_anchor_valid[0, 0])
    assert int(out.proposal_anchor_indices[0, 0]) == 0
    prob = torch.sigmoid(out.mask_logits.detach())
    inside = prob[0, 0, 3:12, 5:18].mean()
    outside = prob[0, 0, :2, :5].mean()
    assert float(inside) > 0.8
    assert float(outside) < 0.2


def test_jpc_up_losses_run_and_backprop():
    torch.manual_seed(1)
    cfg = _tiny_config()
    model = JointPanopticCouplerUp(cfg)
    appearance = torch.randn(1, cfg.appearance_dim, *cfg.low_hw)
    semantic_codes = torch.randn(1, cfg.semantic_dim, *cfg.low_hw)
    depth = torch.rand(1, 1, 32, 64)
    rgb = torch.rand(1, 3, 32, 64)
    out = model(appearance, semantic_codes, depth, rgb)

    semantic_target = torch.randint(0, cfg.num_prototypes, (1, *cfg.high_hw))
    proposal_masks = torch.zeros(1, 3, *cfg.high_hw)
    proposal_masks[:, 0, 2:10, 4:14] = 1.0
    proposal_masks[:, 1, 5:15, 16:28] = 1.0
    proposal_masks[:, 2, 0:4, 0:5] = 1.0
    proposal_scores = torch.tensor([[0.9, 0.7, 0.2]])
    boundary_target = torch.rand(1, 1, *cfg.high_hw)
    center_target = torch.rand(1, 1, *cfg.high_hw)

    losses = compute_jpc_up_losses(
        out,
        semantic_target=semantic_target,
        teacher_codes=semantic_codes,
        proposal_masks=proposal_masks,
        proposal_scores=proposal_scores,
        center_target=center_target,
        boundary_target=boundary_target,
        config=JPCUpLossConfig(min_proposal_score=0.1),
    )

    assert "loss" in losses
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_latent_consensus_targets_and_graph_loss():
    torch.manual_seed(3)
    cfg = _tiny_config()
    model = JointPanopticCouplerUp(cfg)
    out = model(
        torch.randn(1, cfg.appearance_dim, *cfg.low_hw),
        torch.randn(1, cfg.semantic_dim, *cfg.low_hw),
        torch.rand(1, 1, 32, 64),
        torch.rand(1, 3, 32, 64),
    )
    proposal_masks = torch.zeros(1, 2, *cfg.high_hw)
    proposal_masks[:, 0, 2:10, 4:14] = 1.0
    proposal_masks[:, 1, 6:14, 20:30] = 1.0
    proposal_scores = torch.tensor([[0.9, 0.8]])
    sp = torch.arange(cfg.high_hw[0] * cfg.high_hw[1]).reshape(1, *cfg.high_hw) % 16
    edges = [torch.tensor([[i, i + 1] for i in range(15)], dtype=torch.long)]

    builder = LatentConsensusBuilder(
        LatentConsensusConfig(
            output_hw=cfg.high_hw,
            max_consensus_masks=4,
            min_mask_area=4,
            include_model_masks=False,
        )
    )
    targets = builder(
        out,
        proposal_masks=proposal_masks,
        proposal_scores=proposal_scores,
        superpixel_ids=sp,
        edge_indices=edges,
    )

    assert targets.object_masks.shape == (1, 2, *cfg.high_hw)
    assert targets.semantic_target.shape == (1, cfg.num_prototypes, *cfg.high_hw)
    assert targets.objectness_target.shape == (1, 1, *cfg.high_hw)
    assert targets.edge_indices is not None
    assert targets.edge_targets is not None

    losses = compute_jpc_up_losses(
        out,
        semantic_target=targets.semantic_target,
        semantic_valid=targets.valid_mask[:, 0],
        proposal_masks=targets.object_masks,
        proposal_scores=targets.object_scores,
        objectness_target=targets.objectness_target,
        center_target=targets.center_target,
        boundary_target=targets.boundary_target,
        superpixel_ids=sp,
        edge_indices=targets.edge_indices,
        edge_targets=targets.edge_targets,
        edge_weights=targets.edge_weights,
    )
    assert torch.isfinite(losses["L_graph_edge"])
    assert torch.isfinite(losses["loss"])


def test_jpc_up_losses_allow_missing_evidence():
    torch.manual_seed(2)
    cfg = _tiny_config()
    model = JointPanopticCouplerUp(cfg)
    out = model(
        torch.randn(1, cfg.appearance_dim, *cfg.low_hw),
        torch.randn(1, cfg.semantic_dim, *cfg.low_hw),
        torch.rand(1, 1, 32, 64),
        None,
    )

    losses = compute_jpc_up_losses(out)
    assert torch.isfinite(losses["loss"])
    assert float(losses["L_semantic"].detach()) == 0.0
    assert float(losses["L_objectness"].detach()) == 0.0
    assert float(losses["L_proposal_mask"].detach()) == 0.0
    assert float(losses["L_mask_coverage"].detach()) == 0.0
    assert float(losses["L_query_score"].detach()) == 0.0
    assert float(losses["L_direct_proposal_mask"].detach()) == 0.0
    assert float(losses["L_direct_mask_coverage"].detach()) == 0.0
    assert float(losses["L_direct_query_score"].detach()) == 0.0
    assert float(losses["L_mask_background"].detach()) == 0.0
    assert torch.isfinite(losses["L_mask_prob_floor"])


def test_direct_positive_mask_losses_bypass_empty_consensus_masks():
    torch.manual_seed(4)
    cfg = _tiny_config()
    model = JointPanopticCouplerUp(cfg)
    out = model(
        torch.randn(1, cfg.appearance_dim, *cfg.low_hw),
        torch.randn(1, cfg.semantic_dim, *cfg.low_hw),
        torch.rand(1, 1, 32, 64),
        None,
    )
    direct_masks = torch.zeros(1, 1, *cfg.high_hw)
    direct_masks[:, 0, 3:12, 6:20] = 1.0
    direct_scores = torch.ones(1, 1)
    empty_consensus = torch.zeros(1, 0, *cfg.high_hw)
    empty_scores = torch.zeros(1, 0)

    losses = compute_jpc_up_losses(
        out,
        proposal_masks=empty_consensus,
        proposal_scores=empty_scores,
        direct_proposal_masks=direct_masks,
        direct_proposal_scores=direct_scores,
        direct_proposal_class_ids=torch.tensor([[11]]),
        config=JPCUpLossConfig(
            lambda_semantic=0.0,
            lambda_code_keep=0.0,
            lambda_up_down=0.0,
            lambda_up_edge=0.0,
            lambda_up_var=0.0,
            lambda_objectness=0.0,
            lambda_center=0.0,
            lambda_boundary=0.0,
            lambda_proposal_mask=0.0,
            lambda_mask_coverage=0.0,
            lambda_query_score=0.0,
            lambda_direct_proposal_mask=1.0,
            lambda_direct_mask_coverage=1.0,
            lambda_direct_query_score=1.0,
            lambda_query_class=1.0,
            lambda_mask_background=0.0,
            lambda_embedding=0.0,
            lambda_graph_edge=0.0,
            lambda_semantic_purity=0.0,
            lambda_overlap=0.0,
        ),
    )

    assert float(losses["L_proposal_mask"].detach()) == 0.0
    assert float(losses["L_direct_proposal_mask"].detach()) > 0.0
    assert float(losses["L_direct_mask_coverage"].detach()) > 0.0
    assert float(losses["L_query_class"].detach()) > 0.0
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    grad = model.query_decoder.mask_feature.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_anchor_supervision_uses_proposal_conditioned_queries_directly():
    torch.manual_seed(11)
    cfg = _tiny_config()
    model = JointPanopticCouplerUp(cfg)
    proposal_masks = torch.zeros(1, 1, *cfg.high_hw)
    proposal_masks[:, 0, 4:13, 8:22] = 1.0
    proposal_scores = torch.ones(1, 1)
    out = model(
        torch.randn(1, cfg.appearance_dim, *cfg.low_hw),
        torch.randn(1, cfg.semantic_dim, *cfg.low_hw),
        torch.rand(1, 1, 32, 64),
        None,
        proposal_masks,
        proposal_scores,
    )

    losses = compute_jpc_up_losses(
        out,
        direct_proposal_masks=proposal_masks,
        direct_proposal_scores=proposal_scores,
        direct_proposal_class_ids=torch.tensor([[13]]),
        config=JPCUpLossConfig(
            lambda_semantic=0.0,
            lambda_code_keep=0.0,
            lambda_up_down=0.0,
            lambda_up_edge=0.0,
            lambda_up_var=0.0,
            lambda_objectness=0.0,
            lambda_center=0.0,
            lambda_boundary=0.0,
            lambda_proposal_mask=0.0,
            lambda_mask_coverage=0.0,
            lambda_query_score=0.0,
            lambda_direct_proposal_mask=1.0,
            lambda_direct_mask_coverage=0.0,
            lambda_direct_query_score=1.0,
            lambda_query_class=1.0,
            lambda_proposal_residual=0.1,
            lambda_mask_background=0.0,
            lambda_embedding=0.0,
            lambda_graph_edge=0.0,
            lambda_semantic_purity=0.0,
            lambda_overlap=0.0,
        ),
    )

    assert torch.isfinite(losses["loss"])
    assert float(losses["L_direct_proposal_mask"].detach()) > 0.0
    assert float(losses["L_direct_query_score"].detach()) > 0.0
    assert float(losses["L_query_class"].detach()) > 0.0
    assert float(losses["L_proposal_residual"].detach()) > 0.0


def test_query_mask_losses_push_up_collapsed_foreground_logits():
    logits = torch.full((1, 2, 8, 8), -20.0, requires_grad=True)
    proposal_masks = torch.zeros(1, 1, 8, 8)
    proposal_masks[:, 0, 2:6, 2:6] = 1.0
    proposal_scores = torch.ones(1, 1)
    cfg = JPCUpLossConfig(min_proposal_score=0.0)

    loss = proposal_mask_loss(logits, proposal_masks, proposal_scores, cfg)
    loss = loss + mask_coverage_loss(logits, proposal_masks, proposal_scores, cfg)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[:, :, 2:6, 2:6].mean()) < 0.0


def test_query_score_and_background_losses_separate_objects_from_broad_masks():
    proposal_masks = torch.zeros(1, 1, 8, 8)
    proposal_masks[:, 0, 2:6, 2:6] = 1.0
    proposal_scores = torch.ones(1, 1)
    cfg = JPCUpLossConfig(min_proposal_score=0.0)

    mask_logits = torch.full((1, 2, 8, 8), -6.0)
    mask_logits[:, 0, 2:6, 2:6] = 6.0
    query_scores = torch.zeros(1, 2, requires_grad=True)
    score_loss = query_score_loss(query_scores, mask_logits, proposal_masks, proposal_scores, cfg)
    score_loss.backward()

    assert torch.isfinite(score_loss)
    assert query_scores.grad is not None
    assert float(query_scores.grad[0, 0]) < 0.0
    assert float(query_scores.grad[0, 1]) > 0.0

    broad_logits = torch.full((1, 2, 8, 8), 5.0, requires_grad=True)
    bg_loss = mask_background_suppression_loss(broad_logits, proposal_masks, proposal_scores, cfg)
    bg_loss.backward()

    assert torch.isfinite(bg_loss)
    assert broad_logits.grad is not None
    assert float(broad_logits.grad[:, :, :2, :].mean()) > 0.0

    scores = torch.ones(1, 2)
    identical_overlap = overlap_loss(torch.stack([broad_logits.detach()[0, 0], broad_logits.detach()[0, 0]]).unsqueeze(0), scores)
    separated = torch.full((1, 2, 8, 8), -6.0)
    separated[:, 0, :3, :3] = 6.0
    separated[:, 1, 5:, 5:] = 6.0
    separated_overlap = overlap_loss(separated, scores)
    assert float(identical_overlap) > float(separated_overlap)


def test_proposal_mask_loss_prefers_one_to_one_query_matches():
    proposal_masks = torch.zeros(1, 2, 8, 8)
    proposal_masks[:, 0, 1:4, 1:4] = 1.0
    proposal_masks[:, 1, 5:8, 5:8] = 1.0
    proposal_scores = torch.ones(1, 2)
    cfg = JPCUpLossConfig(min_proposal_score=0.0)

    good_logits = torch.full((1, 2, 8, 8), -6.0)
    good_logits[:, 0, 1:4, 1:4] = 6.0
    good_logits[:, 1, 5:8, 5:8] = 6.0
    bad_logits = torch.full((1, 2, 8, 8), -6.0)
    bad_logits[:, 0, 1:4, 1:4] = 6.0
    bad_logits[:, 1, 1:4, 1:4] = 6.0

    good_loss = proposal_mask_loss(good_logits, proposal_masks, proposal_scores, cfg)
    bad_loss = proposal_mask_loss(bad_logits, proposal_masks, proposal_scores, cfg)

    assert torch.isfinite(good_loss)
    assert torch.isfinite(bad_loss)
    assert float(good_loss) < float(bad_loss)


def test_export_can_preserve_cache_proposal_masks():
    semantic_logits = torch.zeros(1, 4, 4, 4)
    semantic_logits[:, 2, 1:3, 1:3] = 5.0
    output = SimpleNamespace(
        semantic_logits=semantic_logits,
        mask_logits=torch.full((1, 2, 4, 4), -8.0),
        query_scores=torch.full((1, 2), -8.0),
    )
    proposal_masks = torch.zeros(1, 1, 4, 4)
    proposal_masks[:, 0, 1:3, 1:3] = 1.0
    proposal_scores = torch.tensor([[0.9]])

    sem, instances, panoptic = output_to_pseudolabels(
        output,
        final_hw=(16, 16),
        score_threshold=0.99,
        include_proposals=True,
        proposal_masks=proposal_masks,
        proposal_scores=proposal_scores,
        proposal_score_threshold=0.1,
        min_area=1,
    )

    assert sem.shape == (16, 16)
    assert len(instances) == 1
    assert instances[0][1] == 2
    assert int((panoptic == 2001).sum()) > 0


def test_export_uses_hard_semantic_majority_for_instance_class_ids():
    semantic_logits = torch.zeros(1, 5, 4, 4)
    semantic_logits[:, 3, 1:3, 1:3] = 3.0
    semantic_logits[:, 2, 1, 1] = 12.0
    output = SimpleNamespace(
        semantic_logits=semantic_logits,
        mask_logits=torch.full((1, 1, 4, 4), -8.0),
        query_scores=torch.tensor([[8.0]]),
    )
    output.mask_logits[:, 0, 1:3, 1:3] = 8.0

    _, instances, panoptic = output_to_pseudolabels(
        output,
        final_hw=(16, 16),
        score_threshold=0.1,
        mask_threshold=0.5,
        min_area=1,
        instance_class_source="semantic_argmax",
    )

    assert len(instances) == 1
    assert instances[0][1] == 3
    assert int((panoptic == 3001).sum()) > 0


def test_export_can_use_query_class_logits_as_trainids():
    semantic_logits = torch.zeros(1, 80, 4, 4)
    output = SimpleNamespace(
        semantic_logits=semantic_logits,
        mask_logits=torch.full((1, 2, 4, 4), -8.0),
        query_scores=torch.tensor([[8.0, 8.0]]),
        query_class_logits=torch.full((1, 2, 20), -8.0),
    )
    output.mask_logits[:, 0, 1:3, 1:3] = 8.0
    output.mask_logits[:, 1, 0:2, 0:2] = 8.0
    output.query_class_logits[:, 0, 13] = 8.0
    output.query_class_logits[:, 1, 19] = 8.0

    _, instances, panoptic = output_to_pseudolabels(
        output,
        final_hw=(16, 16),
        score_threshold=0.1,
        mask_threshold=0.5,
        min_area=1,
        query_class_source="query",
        no_object_class=19,
    )

    assert len(instances) == 1
    assert instances[0][1] == 13
    assert int((panoptic == 13001).sum()) > 0


def test_export_query_class_thing_gate_suppresses_stuff_classes():
    semantic_logits = torch.zeros(1, 80, 4, 4)
    output = SimpleNamespace(
        semantic_logits=semantic_logits,
        mask_logits=torch.full((1, 2, 4, 4), -8.0),
        query_scores=torch.tensor([[8.0, 8.0]]),
        query_class_logits=torch.full((1, 2, 20), -8.0),
    )
    output.mask_logits[:, 0, 1:3, 1:3] = 8.0
    output.mask_logits[:, 1, 0:2, 0:2] = 8.0
    output.query_class_logits[:, 0, 13] = 8.0
    output.query_class_logits[:, 1, 3] = 8.0

    _, instances, _ = output_to_pseudolabels(
        output,
        final_hw=(16, 16),
        score_threshold=0.1,
        mask_threshold=0.5,
        min_area=1,
        query_class_source="query",
        no_object_class=19,
        query_thing_classes={11, 12, 13, 14, 15, 16, 17, 18},
    )

    assert len(instances) == 1
    assert instances[0][1] == 13


def test_export_can_keep_only_anchored_queries_and_use_proposal_scores():
    semantic_logits = torch.zeros(1, 80, 4, 4)
    output = SimpleNamespace(
        semantic_logits=semantic_logits,
        mask_logits=torch.full((1, 3, 4, 4), -8.0),
        query_scores=torch.tensor([[8.0, 8.0, 8.0]]),
        proposal_anchor_valid=torch.tensor([[True, False, True]]),
        proposal_anchor_indices=torch.tensor([[1, -1, 0]]),
    )
    output.mask_logits[:, 0, 1:3, 1:3] = 8.0
    output.mask_logits[:, 1, 0:4, 0:4] = 8.0
    output.mask_logits[:, 2, 0:2, 0:2] = 8.0
    proposal_scores = torch.tensor([[0.25, 0.85]])

    _, instances, _ = output_to_pseudolabels(
        output,
        final_hw=(16, 16),
        score_threshold=0.5,
        mask_threshold=0.5,
        min_area=1,
        proposal_scores=proposal_scores,
        query_score_source="proposal_anchor",
        anchored_query_only=True,
    )

    assert len(instances) == 1
    assert abs(instances[0][2] - 0.85) < 1.0e-6


def test_export_class_aware_nms_keeps_overlapping_different_classes():
    semantic_logits = torch.zeros(1, 80, 4, 4)
    output = SimpleNamespace(
        semantic_logits=semantic_logits,
        mask_logits=torch.full((1, 2, 4, 4), -8.0),
        query_scores=torch.tensor([[9.0, 8.0]]),
        query_class_logits=torch.full((1, 2, 20), -8.0),
    )
    output.mask_logits[:, 0, 0:3, 0:3] = 8.0
    output.mask_logits[:, 1, 1:4, 1:4] = 8.0
    output.query_class_logits[:, 0, 13] = 8.0
    output.query_class_logits[:, 1, 14] = 8.0

    _, instances, panoptic = output_to_pseudolabels(
        output,
        final_hw=(4, 4),
        score_threshold=0.1,
        mask_threshold=0.5,
        min_area=1,
        nms_iou=0.1,
        query_class_source="query",
        no_object_class=19,
        class_aware_nms=True,
    )

    assert len(instances) == 2
    assert [cls for _, cls, _ in instances] == [13, 14]
    assert int((panoptic == 13001).sum()) > 0
    assert int((panoptic == 14002).sum()) > 0


def test_export_max_instances_per_class_caps_same_class_masks():
    semantic_logits = torch.zeros(1, 80, 4, 4)
    output = SimpleNamespace(
        semantic_logits=semantic_logits,
        mask_logits=torch.full((1, 2, 4, 4), -8.0),
        query_scores=torch.tensor([[9.0, 8.0]]),
        query_class_logits=torch.full((1, 2, 20), -8.0),
    )
    output.mask_logits[:, 0, 0:2, 0:2] = 8.0
    output.mask_logits[:, 1, 2:4, 2:4] = 8.0
    output.query_class_logits[:, 0, 13] = 8.0
    output.query_class_logits[:, 1, 13] = 8.0

    _, instances, _ = output_to_pseudolabels(
        output,
        final_hw=(4, 4),
        score_threshold=0.1,
        mask_threshold=0.5,
        min_area=1,
        query_class_source="query",
        no_object_class=19,
        max_instances_per_class=1,
    )

    assert len(instances) == 1
    assert instances[0][1] == 13


def test_export_bank_proposal_rescue_adds_nonduplicate_raw_proposals_after_anchors():
    semantic_logits = torch.zeros(1, 80, 4, 4)
    output = SimpleNamespace(
        semantic_logits=semantic_logits,
        mask_logits=torch.full((1, 1, 4, 4), -8.0),
        query_scores=torch.tensor([[8.0]]),
        objectness_logits=torch.full((1, 1, 4, 4), 4.0),
        center_logits=torch.full((1, 1, 4, 4), 4.0),
    )
    output.mask_logits[:, 0, 0:2, 0:2] = 8.0
    class_map = torch.zeros(1, 4, 4)
    class_map[:, 0:2, 0:2] = 13
    class_map[:, 2:4, 2:4] = 13
    bank_masks = torch.zeros(1, 2, 4, 4)
    bank_masks[:, 0, 0:2, 0:2] = 1.0
    bank_masks[:, 1, 2:4, 2:4] = 1.0
    bank_scores = torch.tensor([[0.95, 0.9]])

    _, instances, panoptic = output_to_pseudolabels(
        output,
        final_hw=(4, 4),
        score_threshold=0.1,
        mask_threshold=0.5,
        min_area=1,
        class_map=class_map,
        query_score_source="query",
        instance_class_source="class_map",
        class_aware_nms=True,
        bank_proposal_masks=bank_masks,
        bank_proposal_scores=bank_scores,
        include_bank_proposal_rescue=True,
        bank_proposal_score_threshold=0.5,
        bank_proposal_min_area=1,
        bank_proposal_max_anchor_iou=0.2,
        bank_proposal_classes={13},
    )

    assert len(instances) == 2
    assert [cls for _, cls, _ in instances] == [13, 13]
    assert int((panoptic == 13001).sum()) == 4
    assert int((panoptic == 13002).sum()) == 4


def test_export_bank_proposal_rescue_can_gate_by_class_and_objectness():
    semantic_logits = torch.zeros(1, 80, 4, 4)
    output = SimpleNamespace(
        semantic_logits=semantic_logits,
        mask_logits=torch.full((1, 0, 4, 4), -8.0),
        query_scores=torch.zeros(1, 0),
        objectness_logits=torch.full((1, 1, 4, 4), -8.0),
        center_logits=torch.full((1, 1, 4, 4), 8.0),
    )
    output.objectness_logits[:, :, 0:2, 0:2] = 8.0
    class_map = torch.zeros(1, 4, 4)
    class_map[:, 0:2, 0:2] = 13
    class_map[:, 2:4, 2:4] = 14
    bank_masks = torch.zeros(1, 2, 4, 4)
    bank_masks[:, 0, 0:2, 0:2] = 1.0
    bank_masks[:, 1, 2:4, 2:4] = 1.0
    bank_scores = torch.tensor([[0.9, 0.9]])

    _, instances, panoptic = output_to_pseudolabels(
        output,
        final_hw=(4, 4),
        min_area=1,
        include_query_masks=False,
        class_map=class_map,
        instance_class_source="class_map",
        bank_proposal_masks=bank_masks,
        bank_proposal_scores=bank_scores,
        include_bank_proposal_rescue=True,
        bank_proposal_score_threshold=0.5,
        bank_proposal_min_area=1,
        bank_proposal_min_objectness=0.5,
        bank_proposal_classes={13},
    )

    assert len(instances) == 1
    assert instances[0][1] == 13
    assert int((panoptic == 13001).sum()) == 4
    assert int((panoptic == 14001).sum()) == 0


def test_export_semantic_component_rescue_adds_class_map_components():
    semantic_logits = torch.zeros(1, 80, 4, 4)
    output = SimpleNamespace(
        semantic_logits=semantic_logits,
        mask_logits=torch.full((1, 0, 4, 4), -8.0),
        query_scores=torch.zeros(1, 0),
        objectness_logits=torch.full((1, 1, 4, 4), -4.0),
    )
    output.objectness_logits[:, :, 0:2, 0:2] = 4.0
    output.objectness_logits[:, :, 2:4, 2:4] = 4.0
    class_map = torch.zeros(1, 4, 4)
    class_map[:, 0:2, 0:2] = 13
    class_map[:, 2:4, 2:4] = 13

    _, instances, panoptic = output_to_pseudolabels(
        output,
        final_hw=(4, 4),
        min_area=1,
        include_query_masks=False,
        class_map=class_map,
        include_semantic_components=True,
        semantic_component_min_area=2,
        semantic_component_max_area=8,
        semantic_component_score_threshold=0.5,
        semantic_component_score_scale=1.0,
        semantic_component_max_components=10,
    )

    assert len(instances) == 2
    assert [cls for _, cls, _ in instances] == [13, 13]
    assert int((panoptic == 13001).sum()) == 4
    assert int((panoptic == 13002).sum()) == 4


def test_export_semantic_component_rescue_can_require_proposal_overlap():
    semantic_logits = torch.zeros(1, 80, 4, 4)
    output = SimpleNamespace(
        semantic_logits=semantic_logits,
        mask_logits=torch.full((1, 0, 4, 4), -8.0),
        query_scores=torch.zeros(1, 0),
        objectness_logits=torch.full((1, 1, 4, 4), 4.0),
    )
    class_map = torch.zeros(1, 4, 4)
    class_map[:, 0:2, 0:2] = 13
    class_map[:, 2:4, 2:4] = 13
    proposal_masks = torch.zeros(1, 1, 4, 4)
    proposal_masks[:, 0, 0:2, 0:2] = 1.0
    proposal_scores = torch.tensor([[0.9]])

    _, instances, panoptic = output_to_pseudolabels(
        output,
        final_hw=(4, 4),
        min_area=1,
        include_query_masks=False,
        class_map=class_map,
        include_semantic_components=True,
        semantic_component_min_area=2,
        semantic_component_max_area=8,
        semantic_component_classes={13},
        semantic_component_gate_masks=proposal_masks,
        semantic_component_gate_scores=proposal_scores,
        semantic_component_gate_min_cover=0.75,
        semantic_component_gate_min_score=0.5,
    )

    assert len(instances) == 1
    assert instances[0][1] == 13
    assert int((panoptic == 13001).sum()) == 4
    assert int((panoptic == 13002).sum()) == 0


def test_cache_builder_loads_proposal_bank_from_split_city_layout(tmp_path):
    root = tmp_path / "proposal_bank"
    path = root / "train" / "aachen" / "aachen_000000_000019.npz"
    path.parent.mkdir(parents=True)
    masks = np.zeros((2, 8, 16), dtype=bool)
    masks[0, 1:5, 2:7] = True
    masks[1, 3:7, 9:14] = True
    np.savez_compressed(path, masks=masks, scores=np.asarray([0.25, 0.95], dtype=np.float32))

    out_masks, out_scores, found = optional_proposal_masks(
        tmp_path,
        [root],
        "train",
        "aachen",
        "aachen_000000_000019",
        (4, 8),
        max_masks=1,
        min_score=0.1,
    )

    assert found == [str(path)]
    assert out_masks is not None
    assert out_scores is not None
    assert out_masks.shape == (1, 4, 8)
    assert abs(float(out_scores[0]) - 0.95) < 1.0e-6
    assert int(out_masks.sum()) > 0


def test_cache_builder_loads_unmore_dense_priors(tmp_path):
    path = tmp_path / "proposal_bank" / "train" / "aachen" / "aachen_000000_000019.npz"
    path.parent.mkdir(parents=True)
    objectness = np.zeros((8, 16), dtype=np.float32)
    center = np.zeros((8, 16), dtype=np.float32)
    boundary = np.zeros((8, 16), dtype=np.float32)
    objectness[1:5, 2:8] = 0.7
    center[3, 5] = 1.0
    boundary[:, 7] = 0.9
    np.savez_compressed(
        path,
        masks=np.zeros((0, 8, 16), dtype=np.uint8),
        scores=np.zeros((0,), dtype=np.float32),
        objectness_prior=objectness,
        center_prior=center,
        boundary_prior=boundary,
    )

    priors = load_bank_prior_maps([str(path)], (4, 8))

    assert set(priors) == {"objectness_prior", "center_prior", "boundary_prior"}
    assert priors["objectness_prior"].shape == (4, 8)
    assert priors["center_prior"].max() > 0.0
    assert priors["boundary_prior"].max() > 0.0


def test_semantic_gate_can_reject_fused_mask_without_boundary_support(tmp_path):
    candidate_dir = tmp_path / "candidates"
    semantic_dir = tmp_path / "semantic"
    prior_dir = tmp_path / "priors"
    for root in (candidate_dir, prior_dir):
        (root / "train" / "aachen").mkdir(parents=True)
    semantic_dir.mkdir(parents=True)

    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True
    np.savez_compressed(
        candidate_dir / "train" / "aachen" / "aachen_000000_000019.npz",
        masks=masks,
        scores=np.asarray([0.9, 0.9], dtype=np.float32),
        num_valid=2,
    )
    semantic = np.full((4, 4), 13, dtype=np.uint8)
    Image.fromarray(semantic).save(semantic_dir / "aachen_000000_000019_leftImg8bit_semantic.png")
    boundary = np.zeros((4, 4), dtype=np.float32)
    boundary[0:2, 0:2] = 0.8
    np.savez_compressed(
        prior_dir / "train" / "aachen" / "aachen_000000_000019.npz",
        objectness_prior=np.ones((4, 4), dtype=np.float32),
        center_prior=np.ones((4, 4), dtype=np.float32),
        boundary_prior=boundary,
    )

    out_masks, _, _, out_classes, stats = gate_one(
        city="aachen",
        stem="aachen_000000_000019",
        split="train",
        candidate_dir=candidate_dir,
        semantic_dir=semantic_dir,
        anchor_dir=None,
        prior_dir=prior_dir,
        cluster_lut=None,
        score_thresh=0.0,
        class_thresholds={},
        min_mask_area=1,
        min_semantic_frac=0.0,
        min_objectness=0.0,
        min_center=0.0,
        min_boundary=0.4,
        boundary_width=1,
        prior_score_weight=0.0,
        nms_iou=0.7,
        max_instances=10,
        class_max_instances={},
        default_class_max=10,
        keep_classes=[13],
        class_vote_mode="majority",
        anchor_score_bonus=0.0,
        anchor_class_thresholds={},
        include_tiny_fragments=False,
        tiny_fragment_source="auto",
        tiny_fragment_min_area=1,
        tiny_fragment_max_area=16,
        tiny_fragment_min_objectness=0.0,
        tiny_fragment_min_center=0.0,
        tiny_fragment_min_boundary=0.0,
        tiny_fragment_max_duplicate_iou=0.2,
        tiny_fragment_max_fragments=10,
        tiny_fragment_classes=[],
        tiny_fragment_class_min_area={},
        tiny_fragment_class_max_area={},
        tiny_fragment_class_min_objectness={},
        tiny_fragment_class_min_center={},
        tiny_fragment_class_min_boundary={},
        tiny_fragment_class_max_fragments={},
    )

    assert out_masks.shape[0] == 1
    assert out_classes.tolist() == [13]
    assert stats["prior_gate_input"] == 2
    assert stats["prior_gate_kept"] == 1


def test_semantic_gate_filters_anchor_bank_by_class_threshold(tmp_path):
    candidate_dir = tmp_path / "candidates"
    anchor_dir = tmp_path / "anchors"
    semantic_dir = tmp_path / "semantic"
    for root in (candidate_dir, anchor_dir):
        (root / "train" / "aachen").mkdir(parents=True)
    semantic_dir.mkdir(parents=True)

    np.savez_compressed(
        candidate_dir / "train" / "aachen" / "aachen_000000_000019.npz",
        masks=np.zeros((0, 4, 4), dtype=bool),
        scores=np.zeros((0,), dtype=np.float32),
        num_valid=0,
    )
    anchor_masks = np.zeros((2, 4, 4), dtype=bool)
    anchor_masks[0, 0:2, 0:2] = True
    anchor_masks[1, 2:4, 2:4] = True
    np.savez_compressed(
        anchor_dir / "train" / "aachen" / "aachen_000000_000019.npz",
        masks=anchor_masks,
        scores=np.asarray([0.53, 0.72], dtype=np.float32),
        class_ids=np.asarray([17, 17], dtype=np.int32),
        num_valid=2,
    )

    out_masks, out_scores, _, out_classes, stats = gate_one(
        city="aachen",
        stem="aachen_000000_000019",
        split="train",
        candidate_dir=candidate_dir,
        semantic_dir=semantic_dir,
        anchor_dir=anchor_dir,
        prior_dir=None,
        cluster_lut=None,
        score_thresh=0.0,
        class_thresholds={},
        min_mask_area=1,
        min_semantic_frac=0.0,
        min_objectness=0.0,
        min_center=0.0,
        min_boundary=0.0,
        boundary_width=1,
        prior_score_weight=0.0,
        nms_iou=0.7,
        max_instances=10,
        class_max_instances={},
        default_class_max=10,
        keep_classes=[17],
        class_vote_mode="majority",
        anchor_score_bonus=0.0,
        anchor_class_thresholds={17: 0.7},
        include_tiny_fragments=False,
        tiny_fragment_source="auto",
        tiny_fragment_min_area=1,
        tiny_fragment_max_area=16,
        tiny_fragment_min_objectness=0.0,
        tiny_fragment_min_center=0.0,
        tiny_fragment_min_boundary=0.0,
        tiny_fragment_max_duplicate_iou=0.2,
        tiny_fragment_max_fragments=10,
        tiny_fragment_classes=[],
        tiny_fragment_class_min_area={},
        tiny_fragment_class_max_area={},
        tiny_fragment_class_min_objectness={},
        tiny_fragment_class_min_center={},
        tiny_fragment_class_min_boundary={},
        tiny_fragment_class_max_fragments={},
        anchor_classes_are_train_ids=True,
    )

    assert out_masks.shape[0] == 1
    assert out_classes.tolist() == [17]
    assert abs(float(out_scores[0]) - 0.72) < 1.0e-6
    assert stats["anchor_kept"] == 1


def test_semantic_gate_adds_objectness_center_tiny_fragment(tmp_path):
    candidate_dir = tmp_path / "candidates"
    semantic_dir = tmp_path / "semantic"
    prior_dir = tmp_path / "priors"
    for root in (candidate_dir, prior_dir):
        (root / "train" / "aachen").mkdir(parents=True)
    semantic_dir.mkdir(parents=True)

    np.savez_compressed(
        candidate_dir / "train" / "aachen" / "aachen_000000_000019.npz",
        masks=np.zeros((0, 4, 4), dtype=bool),
        scores=np.zeros((0,), dtype=np.float32),
        num_valid=0,
    )
    semantic = np.zeros((4, 4), dtype=np.uint8)
    semantic[1:3, 1:3] = 13
    Image.fromarray(semantic).save(semantic_dir / "aachen_000000_000019_leftImg8bit_semantic.png")
    objectness = np.zeros((4, 4), dtype=np.float32)
    objectness[1:3, 1:3] = 0.6
    center = np.zeros((4, 4), dtype=np.float32)
    center[2, 2] = 0.9
    np.savez_compressed(
        prior_dir / "train" / "aachen" / "aachen_000000_000019.npz",
        objectness_prior=objectness,
        center_prior=center,
        boundary_prior=np.ones((4, 4), dtype=np.float32),
    )

    out_masks, out_scores, _, out_classes, stats = gate_one(
        city="aachen",
        stem="aachen_000000_000019",
        split="train",
        candidate_dir=candidate_dir,
        semantic_dir=semantic_dir,
        anchor_dir=None,
        prior_dir=prior_dir,
        cluster_lut=None,
        score_thresh=0.0,
        class_thresholds={},
        min_mask_area=1,
        min_semantic_frac=0.0,
        min_objectness=0.0,
        min_center=0.0,
        min_boundary=0.0,
        boundary_width=1,
        prior_score_weight=0.0,
        nms_iou=0.7,
        max_instances=10,
        class_max_instances={},
        default_class_max=10,
        keep_classes=[13],
        class_vote_mode="majority",
        anchor_score_bonus=0.0,
        anchor_class_thresholds={},
        include_tiny_fragments=True,
        tiny_fragment_source="peak_cc",
        tiny_fragment_min_area=2,
        tiny_fragment_max_area=8,
        tiny_fragment_min_objectness=0.5,
        tiny_fragment_min_center=0.8,
        tiny_fragment_min_boundary=0.0,
        tiny_fragment_max_duplicate_iou=0.2,
        tiny_fragment_max_fragments=10,
        tiny_fragment_classes=[],
        tiny_fragment_class_min_area={},
        tiny_fragment_class_max_area={},
        tiny_fragment_class_min_objectness={},
        tiny_fragment_class_min_center={},
        tiny_fragment_class_min_boundary={},
        tiny_fragment_class_max_fragments={},
    )

    assert out_masks.shape[0] == 1
    assert int(out_masks[0].sum()) == 4
    assert out_classes.tolist() == [13]
    assert float(out_scores[0]) > 0.0
    assert stats["fragments_generated"] == 1


def test_semantic_gate_uses_superpixel_fragments_for_rescue_classes(tmp_path):
    candidate_dir = tmp_path / "candidates"
    semantic_dir = tmp_path / "semantic"
    prior_dir = tmp_path / "priors"
    for root in (candidate_dir, prior_dir):
        (root / "train" / "aachen").mkdir(parents=True)
    semantic_dir.mkdir(parents=True)

    np.savez_compressed(
        candidate_dir / "train" / "aachen" / "aachen_000000_000019.npz",
        masks=np.zeros((0, 4, 4), dtype=bool),
        scores=np.zeros((0,), dtype=np.float32),
        num_valid=0,
    )
    semantic = np.zeros((4, 4), dtype=np.uint8)
    semantic[0:2, 0:2] = 11
    semantic[2:4, 2:4] = 13
    Image.fromarray(semantic).save(semantic_dir / "aachen_000000_000019_leftImg8bit_semantic.png")
    superpixels = np.arange(16, dtype=np.int32).reshape(4, 4)
    superpixels[0:2, 0:2] = 1
    superpixels[2:4, 2:4] = 2
    np.savez_compressed(
        prior_dir / "train" / "aachen" / "aachen_000000_000019.npz",
        objectness_prior=np.ones((4, 4), dtype=np.float32),
        center_prior=np.ones((4, 4), dtype=np.float32),
        boundary_prior=np.ones((4, 4), dtype=np.float32),
        superpixel_ids=superpixels,
    )

    out_masks, _, _, out_classes, stats = gate_one(
        city="aachen",
        stem="aachen_000000_000019",
        split="train",
        candidate_dir=candidate_dir,
        semantic_dir=semantic_dir,
        anchor_dir=None,
        prior_dir=prior_dir,
        cluster_lut=None,
        score_thresh=0.0,
        class_thresholds={},
        min_mask_area=1,
        min_semantic_frac=0.0,
        min_objectness=0.0,
        min_center=0.0,
        min_boundary=0.0,
        boundary_width=1,
        prior_score_weight=0.0,
        nms_iou=0.7,
        max_instances=10,
        class_max_instances={},
        default_class_max=10,
        keep_classes=[11, 13],
        class_vote_mode="majority",
        anchor_score_bonus=0.0,
        anchor_class_thresholds={},
        include_tiny_fragments=True,
        tiny_fragment_source="superpixel",
        tiny_fragment_min_area=2,
        tiny_fragment_max_area=8,
        tiny_fragment_min_objectness=0.5,
        tiny_fragment_min_center=0.5,
        tiny_fragment_min_boundary=0.0,
        tiny_fragment_max_duplicate_iou=0.2,
        tiny_fragment_max_fragments=10,
        tiny_fragment_classes=[11],
        tiny_fragment_class_min_area={},
        tiny_fragment_class_max_area={},
        tiny_fragment_class_min_objectness={},
        tiny_fragment_class_min_center={},
        tiny_fragment_class_min_boundary={},
        tiny_fragment_class_max_fragments={},
    )

    assert out_masks.shape[0] == 1
    assert int(out_masks[0].sum()) == 4
    assert out_classes.tolist() == [11]
    assert stats["fragments_generated"] == 1
    assert stats["fragments_used_superpixels"] == 1


def test_semantic_gate_can_merge_superpixel_fragments_into_same_class_masks(tmp_path):
    candidate_dir = tmp_path / "candidates"
    semantic_dir = tmp_path / "semantic"
    prior_dir = tmp_path / "priors"
    for root in (candidate_dir, prior_dir):
        (root / "train" / "aachen").mkdir(parents=True)
    semantic_dir.mkdir(parents=True)

    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    np.savez_compressed(
        candidate_dir / "train" / "aachen" / "aachen_000000_000019.npz",
        masks=masks,
        scores=np.asarray([0.9], dtype=np.float32),
        num_valid=1,
    )
    semantic = np.zeros((4, 4), dtype=np.uint8)
    semantic[0:2, 0:4] = 11
    Image.fromarray(semantic).save(semantic_dir / "aachen_000000_000019_leftImg8bit_semantic.png")
    superpixels = np.arange(16, dtype=np.int32).reshape(4, 4)
    superpixels[0:2, 0:2] = 1
    superpixels[0:2, 2:4] = 2
    np.savez_compressed(
        prior_dir / "train" / "aachen" / "aachen_000000_000019.npz",
        objectness_prior=np.ones((4, 4), dtype=np.float32),
        center_prior=np.ones((4, 4), dtype=np.float32),
        boundary_prior=np.ones((4, 4), dtype=np.float32),
        superpixel_ids=superpixels,
    )

    out_masks, _, _, out_classes, stats = gate_one(
        city="aachen",
        stem="aachen_000000_000019",
        split="train",
        candidate_dir=candidate_dir,
        semantic_dir=semantic_dir,
        anchor_dir=None,
        prior_dir=prior_dir,
        cluster_lut=None,
        score_thresh=0.0,
        class_thresholds={},
        min_mask_area=1,
        min_semantic_frac=0.0,
        min_objectness=0.0,
        min_center=0.0,
        min_boundary=0.0,
        boundary_width=1,
        prior_score_weight=0.0,
        nms_iou=0.7,
        max_instances=10,
        class_max_instances={},
        default_class_max=10,
        keep_classes=[11],
        class_vote_mode="majority",
        anchor_score_bonus=0.0,
        anchor_class_thresholds={},
        include_tiny_fragments=True,
        tiny_fragment_source="superpixel",
        tiny_fragment_min_area=2,
        tiny_fragment_max_area=8,
        tiny_fragment_min_objectness=0.5,
        tiny_fragment_min_center=0.5,
        tiny_fragment_min_boundary=0.0,
        tiny_fragment_max_duplicate_iou=0.2,
        tiny_fragment_max_fragments=10,
        tiny_fragment_classes=[11],
        tiny_fragment_class_min_area={},
        tiny_fragment_class_max_area={},
        tiny_fragment_class_min_objectness={},
        tiny_fragment_class_min_center={},
        tiny_fragment_class_min_boundary={},
        tiny_fragment_class_max_fragments={},
        tiny_fragment_merge_mode="merge_same_class",
        tiny_fragment_merge_min_touch=0.25,
        tiny_fragment_merge_dilation=1,
    )

    assert out_masks.shape[0] == 1
    assert int(out_masks[0].sum()) == 8
    assert out_classes.tolist() == [11]
    assert stats["fragments_generated"] == 1
    assert stats["fragments_merged"] == 1
    assert stats["fragments_appended"] == 0


def test_rare_thing_scorer_emits_motorcycle_proxy_without_class17_pixels():
    semantic = np.zeros((128, 128), dtype=np.int32)
    semantic[72:88, 42:80] = 12
    semantic[84:98, 42:80] = 18
    mask = np.zeros((128, 128), dtype=bool)
    mask[72:98, 42:80] = True
    box = np.asarray([42, 72, 80, 98], dtype=np.float32)
    priors = {
        "objectness_prior": np.ones((128, 128), dtype=np.float32),
        "center_prior": np.ones((128, 128), dtype=np.float32),
        "boundary_prior": np.ones((128, 128), dtype=np.float32),
    }

    feats = candidate_features(mask, box, semantic, priors, boundary_width=1)
    scores = score_rare_classes(feats, source_score=0.7)

    assert feats.f_motorcycle == 0.0
    assert scores[17] > 0.40
    assert class_is_plausible(17, feats, min_rare_frac=0.08)


def test_rare_thing_anchor_selection_keeps_trainid_classes():
    semantic = np.zeros((8, 8), dtype=np.int32)
    semantic[0:6, 0:3] = 11
    semantic[4:8, 3:8] = 18
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, 0:6, 0:3] = True
    masks[1, 4:8, 3:8] = True
    boxes = np.asarray([[0, 0, 3, 6], [3, 4, 8, 8]], dtype=np.float32)
    scores = np.asarray([0.8, 0.7], dtype=np.float32)
    priors = {
        "objectness_prior": np.ones((8, 8), dtype=np.float32),
        "center_prior": np.ones((8, 8), dtype=np.float32),
        "boundary_prior": np.ones((8, 8), dtype=np.float32),
    }

    out_masks, out_scores, out_boxes, out_classes, stats = select_rare_anchors(
        masks,
        scores,
        boxes,
        semantic,
        priors,
        class_thresholds={11: 0.25, 17: 0.95, 18: 0.25},
        class_max_instances={11: 4, 18: 4},
        min_rare_frac=0.08,
        boundary_width=1,
        allow_multi_class_hypotheses=False,
    )

    assert out_masks.shape[0] == 2
    assert out_scores.shape == (2,)
    assert out_boxes.shape == (2, 4)
    assert sorted(out_classes.tolist()) == [11, 18]
    assert stats["rare_candidates_scored"] == 2
    assert stats["rare_candidates_kept"] == 2


def test_anchor_bank_can_preserve_trainid_classes_with_cluster_lut(tmp_path):
    anchor_dir = tmp_path / "anchors" / "train" / "aachen"
    anchor_dir.mkdir(parents=True)
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True
    np.savez_compressed(
        anchor_dir / "aachen_000000_000019.npz",
        masks=masks,
        scores=np.asarray([0.9], dtype=np.float32),
        class_ids=np.asarray([17], dtype=np.int32),
        num_valid=1,
    )
    cluster_lut = np.full((80,), 255, dtype=np.int32)
    cluster_lut[17] = 3

    _, _, _, mapped_classes = load_anchor_bank(
        tmp_path / "anchors",
        "train",
        "aachen",
        "aachen_000000_000019",
        cluster_lut,
        min_mask_area=1,
        score_thresh=0.0,
    )
    _, _, _, trainid_classes = load_anchor_bank(
        tmp_path / "anchors",
        "train",
        "aachen",
        "aachen_000000_000019",
        cluster_lut,
        min_mask_area=1,
        score_thresh=0.0,
        classes_are_train_ids=True,
    )

    assert mapped_classes.tolist() == [3]
    assert trainid_classes.tolist() == [17]


def test_rare_anchor_base_bank_can_filter_to_thing_trainids(tmp_path):
    root = tmp_path / "anchors"
    out_dir = root / "train" / "aachen"
    out_dir.mkdir(parents=True)
    masks = np.zeros((4, 6, 6), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 0:3, 2:4] = True
    masks[2, 2:5, 0:3] = True
    masks[3, 3:6, 3:6] = True
    boxes = np.asarray([[0, 0, 2, 2], [2, 0, 4, 3], [0, 2, 3, 5], [3, 3, 6, 6]], dtype=np.float32)
    scores = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    class_ids = np.asarray([0, 11, 13, 18], dtype=np.int32)
    np.savez_compressed(
        out_dir / "aachen_000000_000019.npz",
        masks=masks,
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
        num_valid=4,
    )

    out_masks, out_scores, out_boxes, out_classes = load_base_anchors(
        root,
        "train",
        "aachen",
        "aachen_000000_000019",
        cluster_lut=None,
        target_hw=(6, 6),
        min_mask_area=1,
        base_score_bonus=0.05,
        keep_classes=[11, 12, 13, 14, 15, 16, 17, 18],
    )

    assert out_masks.shape[0] == 3
    assert out_scores.shape == (3,)
    assert out_boxes.shape == (3, 4)
    assert sorted(out_classes.tolist()) == [11, 13, 18]
    assert 0 not in out_classes.tolist()
