# Rare-Class Pseudo-Label Recovery: 5 Concrete Proposals for Dead Classes

**Context:** MBPS Stage-3/4 unsupervised panoptic segmentation on Cityscapes. After Stage-4 fine-tuning (Seesaw Loss + class-aware thresholds), 5 classes remain at 0% PQ: **guard rail, tunnel, polegroup, caravan, trailer**. Root cause: these classes occupy <0.02% of pseudo-label pixels — the generator never produces them, so Stage-2 receives zero training signal.

**Pipeline under analysis:**
1. DCFA (Depth-Conditioned Feature Alignment) → overclustered semantics (k=80)
2. DepthPro → depth-guided instance proposals
3. SIMCF-ABC filtering → refines semantic pseudo-labels
4. Thing/stuff split → pseudo-labels for training

---

## Proposal 1: Frequency-Aware Overclustering with Reserved Cluster Budget

### Why it helps

Current k-means overclustering (`generate_overclustered_semantics.py`, k=80) allocates centroids proportionally to feature density. Dominant classes (road, building, vegetation, sidewalk) consume ~54/80 clusters, leaving rare classes to compete for the remaining 26. Guard rail, tunnel, and polegroup are spatially thin and semantically rare; their feature manifolds are swallowed by nearby dense classes (wall/building for guard rail, building for tunnel, pole for polegroup).

The insight from [Liu et al., IDDC, IJCV 2024] is that standard k-means assumes spherical clusters and ignores pixel-class imbalance, causing cluster degeneration. By explicitly reserving a budget of centroids for low-density regions, we force the clustering algorithm to maintain discriminative boundaries for rare classes.

### Implementation

Modify `fit_kmeans()` in `generate_overclustered_semantics.py` (lines 270–348):

```python
def fit_kmeans_frequency_aware(net, segment, cityscapes_root, device, crop_size,
                               patch_size, k=80, rare_budget=12, seed=42):
    """
    Two-stage k-means:
      1. Standard MiniBatchKMeans on all features → k_all = k - rare_budget centroids
      2. Density-based sub-clustering of low-density regions → k_rare = rare_budget centroids
      3. Merge: concatenate centroids, re-assign all features to full k-set
    """
    # Step 1: Extract all 90-dim CAUSE features (same as current)
    all_feats, all_labels = extract_all_features(...)
    norms = np.linalg.norm(all_feats, axis=1, keepdims=True)
    feats_norm = all_feats / np.maximum(norms, 1e-8)

    # Step 1a: Fit dominant-cluster k-means
    kmeans_dense = MiniBatchKMeans(
        n_clusters=k - rare_budget, batch_size=10000, max_iter=300,
        random_state=seed, n_init=3
    )
    kmeans_dense.fit(feats_norm)

    # Step 1b: Identify low-density patches (far from any dense centroid)
    dist_to_dense = kmeans_dense.transform(feats_norm).min(axis=1)  # (N,)
    density_threshold = np.percentile(dist_to_dense, 85)  # bottom 15% density
    rare_mask = dist_to_dense > density_threshold

    # Step 2: Sub-cluster only low-density regions
    rare_feats = feats_norm[rare_mask]
    kmeans_rare = MiniBatchKMeans(
        n_clusters=rare_budget, batch_size=2048, max_iter=300,
        random_state=seed, n_init=5
    )
    kmeans_rare.fit(rare_feats)

    # Step 3: Merge and re-fit assignment
    all_centroids = np.vstack([kmeans_dense.cluster_centers_,
                               kmeans_rare.cluster_centers_])
    # Re-normalize
    all_centroids = all_centroids / (np.linalg.norm(all_centroids, axis=1, keepdims=True) + 1e-8)

    # Build majority-vote mapping on full assignment
    # (cosine similarity + argmax, same as current predict_with_kmeans)
    sim = feats_norm @ all_centroids.T  # (N, k)
    cluster_labels = sim.argmax(axis=1)
    conf = np.zeros((k, NUM_CLASSES), dtype=np.int64)
    for cl, gt in zip(cluster_labels, all_labels):
        if gt < NUM_CLASSES:
            conf[cl, gt] += 1
    cluster_to_class = np.argmax(conf, axis=1).astype(np.uint8)

    return all_centroids, cluster_to_class
```

**Files to modify:**
- `unsupervised-panoptic-segmentation/pseudo_labels/generate_overclustered_semantics.py`
- `mbps_pytorch/generate_overclustered_semantics.py` (sync both copies)

**Hyperparameters:**
- `rare_budget=12` (empirical: 19 classes × 4 clusters/class ≈ 76; 80−12=68 for dense classes)
- `density_threshold=85th percentile` (tune on val: aim for rare_budget clusters to map to guard rail, tunnel, polegroup, caravan, trailer, train, motorcycle)

### Expected Impact

| Class | Baseline PQ | Expected PQ | Rationale |
|-------|-------------|-------------|-----------|
| guard rail | 0.00 | 3–8 | Reserved clusters capture thin roadside features |
| tunnel | 0.00 | 2–5 | Low-density planar structures now have dedicated centroids |
| polegroup | 0.00 | 1–4 | Clustered pole groups separated from single poles |
| caravan | 0.00 | 0–2 | Rare thing; marginal gain without instance boost (see Prop 3) |
| trailer | 0.00 | 0–2 | Same as caravan |

**Overall ΔPQ:** +0.8–1.5 (stuff-heavy)

### Computational Cost

- **Training-time:** ~1.3× standard k-means (two fits + one merge). On 500 val images: ~45s → ~60s.
- **Inference-time:** Identical to baseline (same k=80 centroid lookup).

---

## Proposal 2: Depth-Edge-Aware Semantic Splitting for Planar/Thin Structures

### Why it helps

Guard rail and tunnel have **strong depth signatures**:
- **Guard rail:** Thin vertical structure at road boundary with sharp depth discontinuity (road ↔ guard rail ↔ background). Current depth-guided instance splitting (`generate_depth_guided_instances.py`, Sobel threshold τ=0.05–0.20) is designed for *things* (person, car, bus) and never runs on stuff classes.
- **Tunnel:** Large planar structure with consistent depth gradient (receding into vanishing point) and strong horizontal/vertical planar geometry.

The idea extends [Sick et al., DepthG, ICLR 2024] and [Kim et al., EAGLE, CVPR 2024]: depth gradients encode semantic boundaries. We repurpose depth edges as **semantic splitting cues for stuff classes**, not just instance separation for things.

### Implementation

Create a new script: `generate_depth_edge_semantic_split.py` (or modify `generate_depth_guided_instances.py`):

```python
def depth_edge_semantic_split(semantic, depth, depth_edges,
                              target_classes={'guard_rail': 4, 'tunnel': 9},
                              polegroup_merge=True):
    """
    For each target stuff class:
      1. Detect candidate regions from depth edges + geometric priors
      2. Re-label ambiguous boundary pixels near depth discontinuities
      3. For polegroup: merge nearby pole CCs within depth-smooth regions
    """
    H, W = semantic.shape
    semantic_out = semantic.copy()

    # --- Guard Rail: thin structure at road boundary + depth edge ---
    road_mask = (semantic == 0)
    # Depth edges adjacent to road
    road_dilated = ndimage.binary_dilation(road_mask, iterations=5)
    candidate_gr = road_dilated & depth_edges
    # Filter by aspect ratio (thin horizontal band)
    labeled, n = ndimage.label(candidate_gr)
    for cc_id in range(1, n+1):
        cc = labeled == cc_id
        ys, xs = np.where(cc)
        if len(ys) < 50:
            continue
        h_range = ys.max() - ys.min()
        w_range = xs.max() - xs.min()
        aspect = w_range / max(h_range, 1)
        # Guard rail: long horizontal strip, small vertical extent, near road
        if aspect > 3.0 and h_range < 30:
            # Re-assign to guard rail if currently wall/building/fence
            reclassify_mask = cc & np.isin(semantic, [2, 3, 4])
            semantic_out[reclassify_mask] = target_classes['guard_rail']

    # --- Tunnel: planar receding structure with depth gradient ---
    # Hough transform on depth edges to find vanishing-point converging lines
    from skimage.transform import hough_line
    # Detect lines in depth edge map
    tested_angles = np.linspace(-np.pi/4, np.pi/4, 90)
    h, theta, d = hough_line(depth_edges, theta=tested_angles)
    # Find peaks corresponding to converging lines (tunnel mouth)
    peaks = ...  # threshold on hough accumulator
    # Create tunnel mask from converging line region + depth monotonicity
    tunnel_mask = ...
    # Re-assign building pixels in tunnel mask
    reclassify_tunnel = tunnel_mask & (semantic == 2)  # building → tunnel
    semantic_out[reclassify_tunnel] = target_classes['tunnel']

    # --- Polegroup: merge pole CCs within smooth depth regions ---
    if polegroup_merge:
        pole_mask = (semantic == 5)
        labeled_poles, n_poles = ndimage.label(pole_mask)
        # For each pair of nearby pole CCs, check if depth between them is smooth
        for i in range(1, n_poles+1):
            for j in range(i+1, n_poles+1):
                mi = labeled_poles == i
                mj = labeled_poles == j
                # Bounding box between poles
                bbox_mask = bbox_between(mi, mj)
                depth_var = depth[bbox_mask].var()
                if depth_var < 0.01:  # depth-smooth → likely same polegroup
                    semantic_out[mi | mj] = target_classes.get('polegroup', 5)

    return semantic_out
```

**Files to modify/create:**
- New: `mbps_pytorch/generate_depth_edge_semantic_split.py`
- Integrate into: `mbps_pytorch/generate_panoptic_pseudolabels.py` (Step 2 before thing/stuff assembly)
- Or modify: `unsupervised-panoptic-segmentation/pseudo_labels/generate_depth_guided_instances.py` to also output stuff-class refinements

**Dependencies:** `scikit-image` (for Hough transform)

### Expected Impact

| Class | Baseline PQ | Expected PQ | Rationale |
|-------|-------------|-------------|-----------|
| guard rail | 0.00 | 5–12 | Depth-edge + aspect ratio filter strongly discriminative |
| tunnel | 0.00 | 4–10 | Hough-based planar detection on depth edges |
| polegroup | 0.00 | 3–8 | CC merging under depth smoothness |
| caravan | 0.00 | 0 | N/A (thing, not stuff) |
| trailer | 0.00 | 0 | N/A (thing, not stuff) |

**Overall ΔPQ:** +1.2–2.5 (stuff-heavy; guard rail and tunnel are the biggest PQ ceiling blockers)

### Computational Cost

- **Per-image:** ~80ms additional (Hough transform on 512×1024 depth edges + CC analysis).
- **Full dataset (2975 train + 500 val):** ~4.5 min on CPU, ~1.5 min on GPU if parallelized.

---

## Proposal 3: Transitive Pseudo-Label Propagation via Feature-Space Nearest Neighbors

### Why it helps

Caravan and trailer are **rare things** (visually similar to truck/bus) that appear in only a handful of images. Standard clustering assigns them to truck/bus centroids because their feature density is too low to claim dedicated clusters. However, DINOv3 features are semantically rich: caravan/trailer patches are near truck/bus patches in feature space but form small, tight sub-clusters.

The idea draws from [Iscen et al., Label Propagation for Semi-Supervised Learning, CVPR 2019] and [Zhang et al., Prototypical Pseudo-Label Denoising, CVPR 2021]: we build a **dataset-wide k-NN graph** in DINOv3 feature space and propagate confident pseudo-labels transitively. If a truck patch in image A is confidently labeled, and a caravan patch in image B is its k-NN neighbor, we can transfer the label with a confidence discount.

### Implementation

Create new script: `mbps_pytorch/generate_propagated_pseudolabels.py`

```python
def propagate_rare_class_labels(
    features_dir,          # DINOv3 768D patch features
    semantic_dir,          # current pseudo-labels (k=80 mapped or 19-class)
    output_dir,
    rare_classes=[14, 15, 16, 17, 18],  # truck, bus, train, motorcycle, bicycle
    target_rare=[22, 23],  # caravan, trailer (CAUSE 27-class IDs)
    k_nn=20,
    confidence_threshold=0.6,
    propagation_decay=0.85,
):
    """
    1. Load all patch features and pseudo-labels
    2. For each class c in rare_classes, find patches with high-confidence labels
    3. Build FAISS index over all features
    4. For each confident anchor, query k-NN; if neighbor is currently unlabeled
       or labeled as a sibling class, propagate with decayed confidence
    5. Apply class-specific threshold to accept propagated labels
    """
    import faiss

    # Load all features and labels into memory (or memory-mapped)
    all_feats, all_labels, all_img_ids, all_coords = load_patch_dataset(features_dir, semantic_dir)
    # all_feats: (N_patches, 768) float32, L2-normalized

    # Build FAISS IVF index for fast approximate NN
    d = 768
    quantizer = faiss.IndexFlatIP(d)  # inner product = cosine similarity for L2-normed vectors
    nlist = 100
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(all_feats)
    index.add(all_feats)

    propagated = np.full(len(all_labels), -1, dtype=np.int16)
    prop_conf = np.zeros(len(all_labels), dtype=np.float32)

    for anchor_class in rare_classes:
        # High-confidence anchors for this class
        anchor_mask = (all_labels == anchor_class) & (confidences > confidence_threshold)
        anchor_feats = all_feats[anchor_mask]
        if len(anchor_feats) == 0:
            continue

        # Query k-NN for all anchors
        D, I = index.search(anchor_feats, k_nn)  # D: cosine sim, I: indices

        for anchor_idx, neighbors, sims in zip(np.where(anchor_mask)[0], I, D):
            for nb, sim in zip(neighbors, sims):
                if nb == anchor_idx:
                    continue
                # Only propagate to unlabeled or sibling-class pixels
                if all_labels[nb] not in [-1, anchor_class] + sibling_classes(anchor_class):
                    continue
                # Decay confidence with distance
                conf = confidences[anchor_idx] * propagation_decay * sim
                if conf > prop_conf[nb]:
                    prop_conf[nb] = conf
                    propagated[nb] = anchor_class

    # --- Class-specific re-mapping: truck neighbors → caravan/trailer ---
    # Use a lightweight attribute classifier on DINOv3 features
    # (caravan = truck + elongated body + residential context)
    # Or simpler: heuristic on bbox aspect ratio from instance proposals
    for truck_anchor in np.where(all_labels == 14)[0]:  # truck
        neighbors = ...  # k-NN of truck_anchor
        for nb in neighbors:
            if is_likely_caravan(all_feats[nb], all_coords[nb], all_img_ids[nb]):
                propagated[nb] = 22  # caravan
            elif is_likely_trailer(all_feats[nb], all_coords[nb], all_img_ids[nb]):
                propagated[nb] = 23  # trailer

    # Write propagated labels back to per-image label maps
    merge_propagated_labels(semantic_dir, propagated, all_img_ids, all_coords, output_dir)
```

**Helper `is_likely_caravan()`:** Uses instance proposal aspect ratio (caravan is shorter than truck, more rectangular) + DINOv3 feature similarity to a small set of caravan prototypes extracted from CAUSE 27-class logits (cluster 22).

**Files to modify/create:**
- New: `mbps_pytorch/generate_propagated_pseudolabels.py`
- Depends on: pre-extracted DINOv3 features (`dinov3_features/` directory already exists)
- Integrate into: `mbps_pytorch/scripts/generate_refined_pseudolabels.py` as an optional post-processing step

**Hyperparameters:**
- `k_nn=20`, `propagation_decay=0.85`, `confidence_threshold=0.6`
- FAISS IVF index: nlist=100, nprobe=10 (approximate search, fast)

### Expected Impact

| Class | Baseline PQ | Expected PQ | Rationale |
|-------|-------------|-------------|-----------|
| guard rail | 0.00 | 0–1 | Indirect benefit if wall→guard rail propagation triggers |
| tunnel | 0.00 | 0–1 | Minimal; tunnel features are not near dominant classes in k-NN |
| polegroup | 0.00 | 1–3 | Pole→polegroup grouping via spatial proximity in feature space |
| caravan | 0.00 | 5–15 | Direct truck→caravan propagation; visually similar |
| trailer | 0.00 | 5–15 | Direct truck→trailer propagation; visually similar |

**Overall ΔPQ:** +1.5–3.0 (thing-heavy; recovers the two most visible dead thing classes)

### Computational Cost

- **Index build:** ~2 min for 2975 images × 2048 patches = ~6M vectors on GPU.
- **k-NN query:** ~5 min for all anchors.
- **Total:** ~8 min once; can be cached and reused across experiments.

---

## Proposal 4: Geometric Copy-Paste Augmentation of Rare-Class Prototypes

### Why it helps

If real pseudo-labels don't exist for caravan/trailer/guard rail, we can **synthesize training signal** by copying rare-class prototypes from the few images where they *are* detected (even at low confidence) and pasting them onto other images. This is inspired by [Ghiasi et al., Simple Copy-Paste, CVPR 2021] and [Bai et al., Bidirectional Copy-Paste, CVPR 2023], adapted for the unsupervised setting.

Unlike GAN-based synthesis (infeasible on 1080 Ti), copy-paste requires only:
1. A small bank of rare-class patches (from CAUSE 27-class soft logits where cluster 22/23 has non-zero probability, even if not argmax)
2. Geometric-aware placement (respecting depth scale and scene geometry)

Guard rail, caravan, and trailer have **strong geometric priors**:
- Guard rail: appears at road edge, horizontal, at a specific depth range from camera.
- Caravan/trailer: appear on road, at similar depth scale to cars/trucks.

### Implementation

Create new script: `mbps_pytorch/generate_copypaste_rare_prototypes.py`

```python
def build_rare_class_prototype_bank(
    cityscapes_root,
    cause_logits_dir,   # saved CAUSE 27-class soft logits
    depth_dir,
    rare_classes={
        'guard_rail': {'cause_id': 7, 'min_logit': 0.15},
        'tunnel': {'cause_id': 9, 'min_logit': 0.12},
        'caravan': {'cause_id': 22, 'min_logit': 0.10},
        'trailer': {'cause_id': 23, 'min_logit': 0.10},
    },
    min_area=256,
):
    """
    Scan all images for patches where rare-class CAUSE logit > threshold
    (even if not argmax). Extract image patch + depth patch + mask.
    """
    bank = {name: [] for name in rare_classes}
    for img_path in tqdm(all_images):
        logits = load_cause_logits(img_path)  # (27, H, W)
        depth = load_depth(img_path)          # (H, W)
        img = np.array(Image.open(img_path))  # (H, W, 3)

        for name, cfg in rare_classes.items():
            cid = cfg['cause_id']
            prob_map = torch.softmax(torch.from_numpy(logits), dim=0)[cid].numpy()
            mask = prob_map > cfg['min_logit']
            labeled, n = ndimage.label(mask)
            for cc_id in range(1, n+1):
                cc_mask = labeled == cc_id
                if cc_mask.sum() < min_area:
                    continue
                # Extract bounding patch
                ys, xs = np.where(cc_mask)
                patch = img[ys.min():ys.max()+1, xs.min():xs.max()+1]
                depth_patch = depth[ys.min():ys.max()+1, xs.min():xs.max()+1]
                bank[name].append({
                    'patch': patch,
                    'mask': cc_mask[ys.min():ys.max()+1, xs.min():xs.max()+1],
                    'depth_median': np.median(depth_patch),
                    'source_img': img_path,
                })
    return bank


def paste_prototype_onto_image(target_img, target_depth, prototype,
                               target_class_id, placement_heuristic):
    """
    Place prototype onto target image respecting depth scale.
    For guard rail: place at road edge, scale to target depth.
    For caravan/trailer: place on road plane, scale to nearby car depth.
    """
    # Compute scale factor from depth ratio
    proto_depth = prototype['depth_median']
    # Find road pixels in target
    road_depth = target_depth[target_semantic == 0].median()
    scale = road_depth / proto_depth

    # Resize prototype
    h, w = prototype['patch'].shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    resized_patch = cv2.resize(prototype['patch'], (new_w, new_h))
    resized_mask = cv2.resize(prototype['mask'].astype(np.uint8), (new_w, new_h)) > 0

    # Placement heuristics
    if target_class_id == 7:  # guard rail
        # Place at left/right road edge, y=center of road region
        y_pos = road_center_y
        x_pos = 0 if random.choice([True, False]) else target_img.shape[1] - new_w
    else:  # caravan/trailer
        # Place on road, x=random along road width
        y_pos = road_center_y - new_h // 2
        x_pos = random.randint(road_x_min, road_x_max - new_w)

    # Paste with Poisson blending or simple alpha blending
    target_img[y_pos:y_pos+new_h, x_pos:x_pos+new_w][resized_mask] = \
        resized_patch[resized_mask]
    target_semantic[y_pos:y_pos+new_h, x_pos:x_pos+new_w][resized_mask] = target_class_id

    return target_img, target_semantic
```

**Files to modify/create:**
- New: `mbps_pytorch/generate_copypaste_rare_prototypes.py`
- Integrate into: existing copy-paste augmentation in training pipeline (`mbps_pytorch/data/pseudo_label_dataset.py` or training dataloader)
- Or as pseudo-label augmentation: run before Stage-2 training to expand the pseudo-label dataset

**Hyperparameters:**
- `min_logit=0.10–0.15` (low threshold to capture weak signals)
- `min_area=256` pixels
- `paste_per_image=1–2` rare-class objects
- Poisson blending (optional) for realism

### Expected Impact

| Class | Baseline PQ | Expected PQ | Rationale |
|-------|-------------|-------------|-----------|
| guard rail | 0.00 | 8–18 | Strong geometric prior; road-edge placement is accurate |
| tunnel | 0.00 | 2–5 | Tunnel prototypes rare; limited benefit |
| polegroup | 0.00 | 0–2 | Pole prototypes exist; grouping harder |
| caravan | 0.00 | 10–20 | Car prototypes abundant; caravan is minor variation |
| trailer | 0.00 | 10–20 | Similar to caravan |

**Overall ΔPQ:** +2.0–4.0 (mixed; highest potential for caravan/trailer)

### Computational Cost

- **Prototype bank build:** ~10 min once (scan all images for weak logits).
- **Paste augmentation:** ~50ms per paste operation (resize + blend).
- **Memory:** Bank of ~500–2000 prototype patches (~50 MB).

---

## Proposal 5: Multi-Scale Cluster Ensemble with Depth-Guided Splitting

### Why it helps

Current pipeline uses a single k=80 clustering. At this resolution, rare classes compete with fine-grained variations of dominant classes (e.g., building facade texture variations claim centroids that could belong to tunnel). By combining clusterings at **multiple scales** (k=80, k=150, k=300) and using depth-guided consensus, we can recover rare classes that appear consistently across scales but are swamped by noise at any single scale.

This is inspired by [Hamilton et al., STEGO, arXiv 2022] (multi-scale feature correspondence) and [Ke et al., HSG, CVPR 2022] (hierarchical spectral grouping). The ensemble acts as a **denoising mechanism**: a rare-class pixel must win its cluster assignment in at least two scales (or win in one scale and be depth-consistent) to be retained.

### Implementation

Create new script: `mbps_pytorch/generate_multiscale_cluster_ensemble.py`

```python
def multiscale_cluster_ensemble(
    cityscapes_root,
    scales=[80, 150, 300],
    depth_subdir='depth_spidepth',
    feat_subdir='dinov3_features',
    ensemble_mode='depth_guided_consensus',
):
    """
    1. Fit k-means at each scale (or load pre-fitted)
    2. For each pixel, collect cluster IDs across scales
    3. Map cluster IDs → 19 trainID classes via majority vote (per scale)
    4. Ensemble:
       a. 'majority_vote': plain voting across scales
       b. 'depth_guided_consensus': require depth-smooth regions to agree
       c. 'adaptive_weight': weight by cluster purity (lower entropy = higher weight)
    5. For rare-class pixels that win in any scale but lose ensemble:
       apply depth-guided split test to preserve them.
    """
    # Load or fit centroids for each scale
    centroids = {}
    for k in scales:
        centroids[k] = load_or_fit_kmeans(k, cityscapes_root, feat_subdir)

    for img_path in tqdm(all_images):
        feat = load_dinov3_feature(img_path)  # (2048, 768) or (H, W, 768)
        depth = load_depth(img_path)

        # Assign at each scale
        assignments = {}
        for k in scales:
            cluster_ids = centroids[k].predict(feat)  # (N,)
            mapped = map_clusters_to_trainids(cluster_ids, k)  # (N,) trainIDs
            assignments[k] = mapped

        # Ensemble with depth guidance
        N = len(feat)
        final_label = np.full(N, 255, dtype=np.uint8)

        for i in range(N):
            votes = [assignments[k][i] for k in scales]
            unique, counts = np.unique(votes, return_counts=True)

            if ensemble_mode == 'majority_vote':
                winner = unique[counts.argmax()]
                if counts.max() >= 2:  # agree in at least 2 scales
                    final_label[i] = winner

            elif ensemble_mode == 'depth_guided_consensus':
                # Check depth smoothness in local neighborhood
                local_depth_var = compute_local_depth_var(depth, i, radius=3)
                if local_depth_var < 0.02:
                    # Smooth region: require strict consensus (all scales agree)
                    if len(unique) == 1:
                        final_label[i] = unique[0]
                else:
                    # Discontinuity region: accept majority (allows rare classes at boundaries)
                    winner = unique[counts.argmax()]
                    if counts.max() >= 2:
                        final_label[i] = winner

        # --- Rare-class rescue pass ---
        # If a pixel is assigned to a rare class in the HIGH-resolution scale (k=300)
        # but overruled in ensemble, check if it's in a depth-edge region
        # and if the k=300 cluster has high purity for that rare class.
        for rare_cls in [4, 7, 9, 11, 22, 23]:  # fence, guard_rail, tunnel, polegroup, caravan, trailer
            rare_mask_300 = (assignments[300] == rare_cls)
            overruled = rare_mask_300 & (final_label != rare_cls) & (final_label != 255)
            if overruled.any():
                # Check depth-edge presence (rare classes often at discontinuities)
                depth_edge_mask = sobel_depth_edges(depth)
                rescue = overruled & depth_edge_mask.ravel()
                # Verify cluster purity from val-set mapping
                if cluster_purity_300[rare_cls] > 0.4:
                    final_label[rescue] = rare_cls

        save_label_map(final_label, img_path, output_dir)
```

**Files to modify/create:**
- New: `mbps_pytorch/generate_multiscale_cluster_ensemble.py`
- Reuses: existing k-means fitting logic from `generate_overclustered_semantics.py`
- Integrate into: pseudo-label generation pipeline before `convert_to_cups_format.py`

**Hyperparameters:**
- `scales=[80, 150, 300]`
- `ensemble_mode='depth_guided_consensus'`
- `cluster_purity_threshold=0.4` for rare-class rescue
- `local_depth_var_threshold=0.02`

### Expected Impact

| Class | Baseline PQ | Expected PQ | Rationale |
|-------|-------------|-------------|-----------|
| guard rail | 0.00 | 4–10 | k=300 captures fine rail structures; depth-guided consensus preserves them |
| tunnel | 0.00 | 3–8 | k=300 separates tunnel from building; consensus + rescue recovers it |
| polegroup | 0.00 | 2–6 | k=150/300 separates pole groups from vegetation/building |
| caravan | 0.00 | 3–8 | Multi-scale captures small variations from truck/bus |
| trailer | 0.00 | 3–8 | Same as caravan |

**Overall ΔPQ:** +1.5–3.0 (broad recovery across all 5 dead classes)

### Computational Cost

- **Fitting:** 3× k-means fits (k=80 already done; k=150, k=300 additional). k=300 on 6M patches: ~3 min.
- **Inference:** 3× cluster assignments per image → ~0.3s/image (vs 0.1s baseline).
- **Total dataset:** ~20 min for 3475 images (GPU-accelerated FAISS or scikit-learn).

---

## Summary & Recommended Execution Order

| Proposal | Primary Targets | ΔPQ Estimate | Cost | Novelty | Recommended Priority |
|----------|-----------------|--------------|------|---------|---------------------|
| **1. Frequency-Aware Clustering** | guard rail, tunnel, polegroup | +0.8–1.5 | Low | Medium | **P1** |
| **2. Depth-Edge Semantic Split** | guard rail, tunnel, polegroup | +1.2–2.5 | Low | High | **P1** (parallel with #1) |
| **3. k-NN Label Propagation** | caravan, trailer | +1.5–3.0 | Medium | High | **P2** |
| **4. Geometric Copy-Paste** | caravan, trailer, guard rail | +2.0–4.0 | Medium | High | **P2** (after #3) |
| **5. Multi-Scale Ensemble** | all 5 classes | +1.5–3.0 | Medium | Medium | **P3** (combines with #1) |

### Staged Implementation Plan

**Phase A (Stuff-class recovery, weeks 1–2):**
1. Implement Proposal 1 (frequency-aware k=80) → test on val set, measure cluster-to-class mapping for guard rail/tunnel/polegroup.
2. In parallel, implement Proposal 2 (depth-edge splitting) → run on val set, measure pixel recall for guard rail/tunnel.
3. Ensemble: combine both by running depth-edge split **after** frequency-aware clustering (depth split reclassifies ambiguous boundary pixels within rare-class clusters).

**Phase B (Thing-class recovery, weeks 2–3):**
4. Implement Proposal 3 (k-NN propagation) → build FAISS index on train features, propagate truck→caravan/trailer labels.
5. Implement Proposal 4 (copy-paste) → build prototype bank from CAUSE weak logits, augment training images.
6. Ablate: propagation only vs. copy-paste only vs. both.

**Phase C (Polish & ensemble, week 4):**
7. Implement Proposal 5 (multi-scale ensemble) → fit k=150, k=300, run depth-guided consensus.
8. Final pipeline: Frequency-aware k=80 → Depth-edge split → Multi-scale rescue → k-NN propagation for things → Copy-paste augmentation.
9. Full Stage-2 retrain → Stage-4 Seesaw fine-tune → measure PQ on all 19 classes.

### Key Papers to Cite

1. **Liu et al.** — *Imbalance-Aware Dense Discriminative Clustering for Unsupervised Semantic Segmentation*, IJCV 2024. (Cluster degeneration, Weibull regularizer)
2. **Sick et al.** — *DepthG: Depth-Guided Correlations for Unsupervised Semantic Segmentation*, ICLR 2024. (Depth-semantic alignment)
3. **Iscen et al.** — *Label Propagation for Deep Semi-Supervised Learning*, CVPR 2019. (Transitive label propagation)
4. **Ghiasi et al.** — *Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation*, CVPR 2021. (Copy-paste augmentation)
5. **Hamilton et al.** — *STEGO: Unsupervised Semantic Segmentation by Distilling Feature Correspondences*, ICLR 2022. (Multi-scale feature correspondence)
6. **Lee et al.** — *ABC: Auxiliary Balanced Classifier for Class-Imbalanced Semi-Supervised Learning*, NeurIPS 2021. (Class-imbalanced pseudo-labeling)
7. **Yu et al.** — *InPL: Pseudo-Labeling the Inliers First for Imbalanced Semi-Supervised Learning*, ICLR 2023. (Imbalanced pseudo-label filtering)

---

*Report generated for MBPS project rare-class recovery track.*
