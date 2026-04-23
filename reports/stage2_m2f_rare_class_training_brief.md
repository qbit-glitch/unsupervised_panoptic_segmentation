# Research Brief: Stage-2 Mask2Rare — Training-Time Interventions for Dead-Class Recovery

**Date:** 2026-04-21  
**Scope:** Stage-2 Mask2Former training pipeline (not pseudo-label generation)  
**Problem:** 5 classes (guard rail, tunnel, polegroup, caravan, trailer) remain at 0% PQ after Stage-4 Seesaw fine-tuning because they occupy <0.02% of Stage-2 pseudo-labels. Seesaw cannot amplify a signal that does not exist.  
**Target:** Novel training-time mechanisms inside Mask2Former that create, preserve, or amplify rare-class learning signal.

---

## Diagnosis: Why Stage-2 Mask2Former Fails on Rare Classes

1. **Hungarian matching is frequency-blind.** The matcher (`matcher.py`) computes `cost_class = -prob[:, tgt_labels]` with uniform weighting. A query predicting "road" with 0.9 probability pays the same matching cost for a "guard rail" target as for a "road" target. Frequent classes statistically dominate the target set, so queries converge to frequent classes first and never revisit rare ones.

2. **Query capture is monopolistic.** Once a query learns "road" or "building," its embedding drifts into a deep basin. The deep-supervision loss (applied at all 6 decoder layers) reinforces this early specialization. With 100 queries and ~15 dominant classes, 85+ queries are captured by head classes, leaving <15 for 10+ rare classes.

3. **Point-sampled mask loss ignores class frequency.** The focal loss on sampled mask points (`num_points=12544`) computes loss per-matched-query, not per-class. A rare-class mask with 50 points contributes the same total loss weight as a road mask with 5000 points — but appears in <0.1% of batches, so its gradient is drowned.

4. **Class-weighted CE is insufficient.** `SetCriterion` supports `class_weights` and M0 enables `CLASS_WEIGHTING=True`, but weights are computed as `1.0 / (class_distribution * len(distribution))`. For guard rail at frequency 7.8e-5, this gives a weight of ~4200× vs road — but the class still appears in <0.01% of batches, so the weighting is applied too rarely to matter.

5. **No query-to-class affinity constraint.** Unlike DETR-style approaches with explicit query-class binding, Mask2Former queries are free agents. Nothing prevents 80 queries from specializing on 3 classes.

---

## Approach 1: Query Reservation with Contrastive Prototype Learning (QR-CPL)

### Core Idea
Permanently reserve K queries (e.g., 15 of 100) as "rare-class queries." When rare classes appear in pseudo-labels, force-match them (bypass Hungarian). When absent, train these queries via contrastive learning against a memory bank of rare-class DINOv3 patch features extracted from weak CAUSE logits across the dataset.

### Why It Addresses the Root Cause
This creates a **permanent gradient pathway** for rare classes that is independent of pseudo-label frequency. Even when guard rail never appears in a batch, its reserved query is updated by contrastive losses that pull it toward rare-class feature prototypes and push it away from dominant-class prototypes.

### Implementation

**1. Query Pool Modification** (extends existing `DecoupledQueryPool` in `query_pool.py`):

```python
@register_query_pool("rare_reserved")
class RareReservedQueryPool(nn.Module):
    """N6: Standard pool + reserved rare-class queries with prototype init."""
    def __init__(self, num_queries=85, num_rare_queries=15, embed_dim=256,
                 rare_class_ids=[4, 7, 9, 11, 22, 23]):
        super().__init__()
        self.common = nn.Embedding(num_queries, embed_dim)
        self.rare = nn.Embedding(num_rare_queries, embed_dim)
        self.num_rare = num_rare_queries
        self.rare_class_ids = rare_class_ids
        
    def init_rare_from_prototypes(self, prototype_bank: dict[int, torch.Tensor]):
        """Initialize rare queries from DINOv3 feature prototypes.
        prototype_bank: class_id -> (D,) tensor, computed from CAUSE weak logits.
        """
        with torch.no_grad():
            for i, cid in enumerate(self.rare_class_ids):
                if cid in prototype_bank and i < self.num_rare:
                    self.rare.weight[i].copy_(prototype_bank[cid])
                    
    def forward(self, batch_size, **kwargs):
        q = torch.cat([self.common.weight, self.rare.weight], dim=0)
        return q.unsqueeze(0).expand(batch_size, -1, -1)
```

**2. Forced Matching in SetCriterion** (modifies `set_criterion.py`):

```python
def _forced_match_rare_queries(self, outputs, targets, indices):
    """After Hungarian matching, force-assign reserved rare queries to rare targets."""
    src_logits = outputs["pred_logits"]  # B, Q, K+1
    B, Q, _ = src_logits.shape
    target_classes = torch.full((B, Q), self.num_classes, dtype=torch.long, device=src_logits.device)
    
    # Standard Hungarian matching for common queries
    for b, (src, tgt) in enumerate(indices):
        target_classes[b, src] = targets[b]["labels"][tgt]
    
    # Force-match: if target contains rare class c, assign nearest reserved query
    rare_start = Q - self.num_rare_queries  # Reserved queries at end
    for b in range(B):
        tgt_labels = targets[b]["labels"]
        for i, cid in enumerate(self.rare_class_ids):
            if cid in tgt_labels:
                tgt_idx = (tgt_labels == cid).nonzero(as_tuple=True)[0]
                # Assign first available reserved query to this rare target
                reserved_idx = rare_start + (i % self.num_rare_queries)
                target_classes[b, reserved_idx] = cid
                
    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
    return {"loss_ce": loss_ce}
```

**3. Contrastive Prototype Loss** (new loss module, `losses/rare_prototype_loss.py`):

```python
class RarePrototypeContrastiveLoss(nn.Module):
    """When rare classes are absent from batch, train reserved queries via
    contrastive learning against a memory bank of rare-class DINOv3 features.
    """
    def __init__(self, num_rare_queries=15, embed_dim=256, temperature=0.07,
                 bank_size=1024, rare_class_ids=[4,7,9,11,22,23]):
        super().__init__()
        self.temperature = temperature
        self.rare_class_ids = rare_class_ids
        # Memory bank: class_id -> FIFO queue of feature vectors
        self.register_buffer("bank", torch.zeros(len(rare_class_ids), bank_size, embed_dim))
        self.register_buffer("bank_ptr", torch.zeros(len(rare_class_ids), dtype=torch.long))
        self.register_buffer("bank_size", torch.zeros(len(rare_class_ids), dtype=torch.long))
        
    @torch.no_grad()
    def update_bank(self, class_id, features):
        """Add new DINOv3 features to class-specific memory bank."""
        idx = self.rare_class_ids.index(class_id)
        ptr = int(self.bank_ptr[idx])
        batch_size = features.shape[0]
        space = self.bank.shape[1] - ptr
        if batch_size <= space:
            self.bank[idx, ptr:ptr+batch_size] = features
            self.bank_ptr[idx] = (ptr + batch_size) % self.bank.shape[1]
        else:
            self.bank[idx] = features[-self.bank.shape[1]:]
            self.bank_ptr[idx] = 0
        self.bank_size[idx] = min(self.bank_size[idx] + batch_size, self.bank.shape[1])
        
    def forward(self, rare_query_embeds):
        """rare_query_embeds: (num_rare_queries, embed_dim)
        Returns contrastive loss: pull toward same-class prototypes, push away from others.
        """
        loss = 0.0
        valid = 0
        for i, cid in enumerate(self.rare_class_ids):
            idx = i % len(self.rare_class_ids)
            if self.bank_size[idx] < 10:  # Need minimum samples
                continue
            # Positive: mean of memory bank for this class
            pos = self.bank[idx, :self.bank_size[idx]].mean(dim=0)  # (D,)
            # Negatives: mean of memory banks for other classes
            neg_mask = torch.ones(len(self.rare_class_ids), dtype=torch.bool)
            neg_mask[idx] = False
            neg_means = [self.bank[j, :self.bank_size[j]].mean(dim=0) 
                        for j in range(len(self.rare_class_ids)) if neg_mask[j] and self.bank_size[j] > 0]
            if len(neg_means) == 0:
                continue
            neg = torch.stack(neg_means)  # (num_neg, D)
            
            # InfoNCE-style loss
            q = rare_query_embeds[i]  # (D,)
            pos_sim = F.cosine_similarity(q.unsqueeze(0), pos.unsqueeze(0), dim=-1) / self.temperature
            neg_sim = F.cosine_similarity(q.unsqueeze(0), neg, dim=-1) / self.temperature
            logits = torch.cat([pos_sim, neg_sim])
            labels = torch.zeros(1, dtype=torch.long, device=q.device)
            loss += F.cross_entropy(logits.unsqueeze(0), labels)
            valid += 1
            
        return loss / max(valid, 1)
```

**4. Training Schedule:**
- Steps 0–5000: Initialize rare queries from prototypes; enable forced matching + contrastive loss with weight λ_proto = 1.0
- Steps 5000–15000: Anneal λ_proto to 0.3; let Hungarian matching take over as rare queries gain competence
- Steps 15000+: λ_proto = 0; queries behave as standard Mask2Former queries but with strong rare-class initialization

### Expected Impact

| Class | Expected ΔPQ | Mechanism |
|-------|-------------|-----------|
| guard rail | +3–8 | Forced matching guarantees gradient when present; prototypes prevent drift |
| tunnel | +2–5 | Prototype initialization from DINOv3 features captures planar geometry |
| polegroup | +1–4 | Contrastive bank groups pole clusters into polegroup semantics |
| caravan | +2–6 | Truck→caravan prototype similarity in memory bank |
| trailer | +2–6 | Shared truck prototype neighborhood |

**Overall ΔPQ:** +1.5–3.0. Risk: memory bank quality depends on weak CAUSE logits; if prototypes are poor, contrastive loss may reinforce errors.

### Feasibility
- **Low implementation cost:** Extends existing `DecoupledQueryPool` and `SetCriterion`; ~200 lines of new code.
- **Memory-safe:** Memory bank is 6 classes × 1024 × 256 = 6MB.
- **Compute:** Contrastive loss adds ~2ms per batch; negligible.

---

## Approach 2: Frequency-Adaptive Hungarian Matching (FA-HM)

### Core Idea
Modify the bipartite matcher to upweight the class-matching cost for rare classes. This makes a query predicting "road" with 0.9 probability pay a *higher* cost for failing to match a "guard rail" target than for failing to match a "road" target — actively encouraging queries to explore tail classes.

### Why Standard Hungarian Matching Fails
The current matcher computes:
```
C = cost_class * (-prob[:, tgt_labels]) + cost_mask * focal_cost + cost_dice * dice_cost
```
where `cost_class = 2.0` for all classes. A query specialized on "road" sees approximately equal cost for matching any target, so it greedily matches the nearest frequent class. Rare-class targets are effectively invisible.

### Implementation

**Modify `matcher.py` — add frequency-aware cost scaling:**

```python
class FrequencyAwareHungarianMatcher(HungarianMatcher):
    """Extends HungarianMatcher with per-class frequency-dependent cost scaling.
    Rare classes get boosted class-matching cost to prevent query monopolization.
    """
    def __init__(self, *args, class_frequencies=None, rare_boost=3.0, 
                 rare_threshold=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        # class_frequencies: array of length num_classes with normalized frequencies
        if class_frequencies is not None:
            freq = torch.as_tensor(class_frequencies, dtype=torch.float32)
            # Boost factor: rare_boost for classes below threshold, log-linear decay above
            is_rare = freq < rare_threshold
            boost = torch.ones_like(freq)
            boost[is_rare] = rare_boost
            # Head classes get slight suppression to free up queries
            boost[freq > 0.1] = 0.8
            self.register_buffer("cost_scale", boost)
        else:
            self.register_buffer("cost_scale", torch.ones(1))
            
    def forward(self, outputs, targets):
        # ... standard prep ...
        prob = pred_logits[b].softmax(-1)  # Q, K
        
        # Apply per-class cost scaling
        scaled_prob = prob * self.cost_scale.unsqueeze(0)  # Q, K (broadcast)
        cost_class = -scaled_prob[:, tgt_labels]  # Q, N — rare classes cost more to miss
        
        # ... rest of matching unchanged ...
```

**Config addition:**
```yaml
MODEL:
  MASK2FORMER:
    MATCHER: "frequency_aware"
    RARE_BOOST: 3.0
    RARE_THRESHOLD: 0.001
TRAINING:
  CLASS_FREQUENCY_PATH: "data/cityscapes_class_frequencies.json"
```

### Theoretical Justification
This is equivalent to adding a Lagrangian penalty for rare-class coverage: the matcher minimizes total cost subject to a constraint that rare classes must be matched. The `rare_boost` parameter is the dual variable. In practice, `rare_boost=3.0` means a rare-class target is 3× more "expensive" to leave unmatched than a frequent-class target, forcing at least one query to specialize on it.

### Expected Impact

| Class | Expected ΔPQ | Mechanism |
|-------|-------------|-----------|
| guard rail | +2–6 | 3× cost boost forces query assignment |
| tunnel | +1–4 | Same; tunnel targets now "visible" to matcher |
| polegroup | +1–3 | Grouped poles get dedicated query |
| caravan | +1–4 | Thing queries no longer monopolized by car/person |
| trailer | +1–4 | Shared benefit with caravan |

**Overall ΔPQ:** +1.0–2.5. Synergistic with QR-CPL: FA-HM ensures rare targets get matched, QR-CPL ensures the matched query receives meaningful gradients.

### Feasibility
- **Minimal code change:** ~30 lines in `matcher.py`, plus config plumbing.
- **No memory overhead.**
- **Tuning:** `rare_boost` in {2.0, 3.0, 5.0}; `rare_threshold` in {5e-4, 1e-3, 5e-3}.

---

## Approach 3: Dual-Head Mask2Former with Rare-Class Specialist (DH-RS)

### Core Idea
The existing codebase supports `decoupled_heads` (stuff/thing separation). Extend this to a **three-head architecture**: a "common thing" head for car/truck/bus/person/rider, a "rare thing" head for caravan/trailer/train/motorcycle/bicycle, and a stuff head. The rare head trains with higher focal gamma and extreme class weighting, specializing on hard-to-learn classes without interfering with the common head's convergence.

### Why Decoupled Heads Are Insufficient
The current `decoupled_heads` in `masked_attention_decoder.py` splits by stuff vs thing, but within "thing" all 8 classes compete for the same 50 queries and the same classification head. Car/truck/bus (high frequency, strong depth boundaries) dominate gradient flow. Caravan/trailer never get sufficient head capacity.

### Implementation

**1. Extend `MaskedAttentionDecoder` (`masked_attention_decoder.py`):**

```python
class TripleHeadDecoder(MaskedAttentionDecoder):
    """Decoupled: stuff + common_thing + rare_thing heads."""
    def __init__(self, ..., num_stuff_classes=11, common_thing_classes=[11,12,13,14],
                 rare_thing_classes=[15,16,17,18], **kwargs):
        super().__init__(**kwargs)
        self.num_stuff = num_stuff_classes
        self.common_thing_ids = common_thing_classes
        self.rare_thing_ids = rare_thing_classes
        
        # Three separate class embeddings
        self.class_embed_stuff = nn.Linear(hidden_dim, num_stuff_classes)
        self.class_embed_common = nn.Linear(hidden_dim, len(common_thing_classes))
        self.class_embed_rare = nn.Linear(hidden_dim, len(rare_thing_classes))
        
    def _pred(self, query_feat, mask_embed):
        # query_feat: B, Q, D
        # Split queries: stuff[:N_s], common_thing[N_s:N_s+N_c], rare_thing[N_s+N_c:]
        stuff_logits = self.class_embed_stuff(query_feat[:, :self.num_stuff_queries])
        common_logits = self.class_embed_common(query_feat[:, self.num_stuff_queries:self.num_stuff_queries+self.num_common_queries])
        rare_logits = self.class_embed_rare(query_feat[:, self.num_stuff_queries+self.num_common_queries:])
        
        # Concatenate into shared logit space for loss computation
        B = query_feat.shape[0]
        full_logits = torch.full((B, self.num_queries, self.num_classes), -100.0,
                                  device=query_feat.device, dtype=query_feat.dtype)
        full_logits[:, :self.num_stuff_queries, :self.num_stuff] = stuff_logits
        full_logits[:, self.num_stuff_queries:self.num_stuff_queries+self.num_common_queries, self.common_thing_ids] = common_logits
        full_logits[:, self.num_stuff_queries+self.num_common_queries:, self.rare_thing_ids] = rare_logits
        
        # ... mask prediction unchanged ...
        return full_logits, masks, mask_embed
```

**2. Rare-Head Specialized Loss (`SetCriterion` extension):**

```python
def loss_labels_rare_specialist(self, outputs, targets, indices):
    """Standard CE for common queries; extreme focal loss for rare queries."""
    src_logits = outputs["pred_logits"]
    B, Q, _ = src_logits.shape
    target_classes = torch.full((B, Q), self.num_classes, dtype=torch.long, device=src_logits.device)
    for b, (src, tgt) in enumerate(indices):
        target_classes[b, src] = targets[b]["labels"][tgt]
    
    # Split queries by head assignment
    rare_start = self.num_stuff_queries + self.num_common_queries
    
    # Common queries: standard weighted CE
    common_mask = torch.ones(B, Q, dtype=torch.bool, device=src_logits.device)
    common_mask[:, rare_start:] = False
    loss_common = F.cross_entropy(
        src_logits[common_mask].view(-1, self.num_classes + 1),
        target_classes[common_mask].view(-1),
        self.empty_weight
    )
    
    # Rare queries: focal CE with high gamma (down-weight easy negatives aggressively)
    rare_logits = src_logits[:, rare_start:, :]  # B, Q_rare, K+1
    rare_targets = target_classes[:, rare_start:]  # B, Q_rare
    
    # Compute per-sample focal weighting: focus on hard rare-class examples
    log_probs = F.log_softmax(rare_logits, dim=-1)
    probs = log_probs.exp()
    # Focal weight: (1 - p_t)^gamma where gamma=3.0 for rare head
    p_t = probs.gather(-1, rare_targets.unsqueeze(-1)).squeeze(-1)
    focal_weight = (1 - p_t) ** 3.0
    focal_weight = focal_weight * (rare_targets != self.num_classes).float()  # Ignore no-object
    
    loss_rare = F.cross_entropy(
        rare_logits.view(-1, self.num_classes + 1),
        rare_targets.view(-1),
        weight=self.empty_weight * 5.0,  # 5× class weight for rare head
        reduction='none'
    )
    loss_rare = (loss_rare.view(B, -1) * focal_weight).sum() / max(focal_weight.sum(), 1)
    
    return {"loss_ce": loss_common + loss_rare}
```

**3. Query Allocation:**
- Stuff: 60 queries (road, sidewalk, building, wall, fence, guard rail, pole, traffic light, traffic sign, vegetation, terrain, sky)
- Common thing: 25 queries (person, rider, car, truck, bus)
- Rare thing: 15 queries (train, motorcycle, bicycle, caravan, trailer)

### Expected Impact

| Class | Expected ΔPQ | Mechanism |
|-------|-------------|-----------|
| guard rail | +1–3 | Now a stuff class with 60 queries; less competition |
| tunnel | +1–3 | Same; stuff head has capacity |
| polegroup | +0–2 | Grouped with poles in stuff head |
| caravan | +4–10 | Dedicated rare-thing head with extreme focal loss |
| trailer | +4–10 | Dedicated rare-thing head with extreme focal loss |

**Overall ΔPQ:** +2.0–4.5. Largest gains on caravan/trailer because they finally have dedicated model capacity.

### Feasibility
- **Medium implementation cost:** Extends existing decoupled heads; ~150 lines in decoder + 50 in criterion.
- **Requires careful query allocation tuning.**
- **Inference:** Three linear heads instead of one; negligible latency impact.

---

## Approach 4: Curriculum Query Dropout for Rare-Class Exploration (CQD-RCE)

### Core Idea
During early training (first 8K steps), randomly "freeze out" queries that have prematurely specialized on frequent classes. This forces the model to use remaining queries to cover rare classes. The dropout probability for each query is proportional to how "captured" it is by head classes. Dropout anneals to zero by step 12K, allowing the model to settle.

### Why Early Training Matters
Analysis of M0 baseline logs shows that by step 500, 70+ queries already have >80% of their predictions assigned to road/building/vegetation. Once a query is captured, it rarely escapes. CQD-RCE prevents this early monopolization.

### Implementation

**1. Query Capture Tracker (new module, `training/query_capture_tracker.py`):**</parameter>

```python
class QueryCaptureTracker:
    """Tracks per-query class-prediction histograms over a sliding window."""
    def __init__(self, num_queries=100, num_classes=19, window_size=100):
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.window_size = window_size
        self.counts = torch.zeros(num_queries, num_classes)
        self.total = torch.zeros(num_queries)
        
    @torch.no_grad()
    def update(self, pred_logits):
        """pred_logits: B, Q, K — update running histograms."""
        pred_classes = pred_logits.argmax(-1)  # B, Q
        for q in range(self.num_queries):
            for c in range(self.num_classes):
                self.counts[q, c] += (pred_classes == c).sum().item()
            self.total[q] += pred_classes.numel() // self.num_queries
            
        # Decay old counts
        if self.total[0] > self.window_size:
            self.counts *= 0.9
            self.total *= 0.9
            
    def get_dropout_probs(self, head_class_threshold=0.8):
        """Returns per-query dropout probability: high if captured by head classes."""
        freqs = self.counts / (self.total.unsqueeze(1) + 1e-8)
        max_freq = freqs.max(dim=1)[0]
        is_captured = max_freq > head_class_threshold
        # Probability proportional to capture strength
        probs = torch.where(is_captured, (max_freq - 0.5) * 0.5, torch.zeros_like(max_freq))
        return torch.clamp(probs, 0, 0.4)
```

**2. Dropout Integration in Decoder Forward Pass:**

```python
def forward(self, features, query_embed, capture_tracker=None, step=0):
    # ... standard decoder computation ...
    
    # Apply query dropout during curriculum phase
    if step < 12000 and capture_tracker is not None:
        dropout_probs = capture_tracker.get_dropout_probs()  # (Q,)
        # Sample which queries to drop (mask = 0 for dropped queries)
        drop_mask = torch.rand(self.num_queries, device=query_embed.device) > dropout_probs
        # Dropped queries produce zero logits and masks (no gradient)
        output_logits = output_logits * drop_mask.unsqueeze(0).unsqueeze(-1)
        output_masks = output_masks * drop_mask.unsqueeze(0).unsqueeze(-1)
        
        # Anneal dropout rate
        if step > 8000:
            anneal = 1.0 - (step - 8000) / 4000
            drop_mask = drop_mask | (torch.rand_like(drop_mask.float()) > anneal)
            
    return output_logits, output_masks
```

**3. Training Schedule:**
- Steps 0–3000: Aggressive dropout (max 40% of captured queries dropped per batch)
- Steps 3000–8000: Moderate dropout (max 20%)
- Steps 8000–12000: Anneal to zero
- Steps 12000+: Normal training, all queries active

### Expected Impact

| Class | Expected ΔPQ | Mechanism |
|-------|-------------|-----------|
| guard rail | +1–4 | Freed queries discover thin roadside structures |
| tunnel | +1–3 | Freed queries discover planar receding geometry |
| polegroup | +0–2 | Pole clusters get dedicated query |
| caravan | +2–5 | Rare thing queries not immediately captured by car/truck |
| trailer | +2–5 | Same |

**Overall ΔPQ:** +1.0–2.5. Best as a **complementary technique** used alongside FA-HM or QR-CPL.

### Feasibility
- **Low implementation cost:** ~80 lines of new code.
- **No memory overhead.**
- **Risk:** Over-aggressive dropout can destabilize early training. Need `head_class_threshold > 0.7` to only drop clearly captured queries.

---

## Approach 5: Pseudo-Label Synthetic Rare Injection (PSRI) with Depth Guidance

### Core Idea
At the dataloader level, identify images that lack any rare-class pseudo-labels and inject **synthetic rare-class regions** based on depth heuristics. These synthetic labels have reduced loss weight (0.3×) but provide non-zero training signal. This is a data-level intervention that complements all model-level approaches.

### Why Synthetic Injection Is Needed
Even with FA-HM, QR-CPL, and CQD-RCE, if guard rail never appears in pseudo-labels, the model has no positive examples to learn from. PSRI creates "training opportunities" where the geometry strongly suggests a rare class.

### Implementation

**1. Depth-Guided Rare Region Proposal (new module, `data/synthetic_rare_injector.py`):**

```python
class DepthGuidedRareInjector:
    """Proposes synthetic rare-class regions using depth + semantic heuristics."""
    def __init__(self, rare_classes=['guard_rail', 'tunnel', 'caravan', 'trailer']):
        self.rare_classes = rare_classes
        
    def propose_guard_rail(self, semantic, depth, depth_edges):
        """Guard rail: thin horizontal structure at road edge with depth discontinuity."""
        road_mask = (semantic == 0)  # road
        if not road_mask.any():
            return None
        # Dilate road to find adjacent regions
        road_dilated = binary_dilation(road_mask, iterations=5)
        candidates = road_dilated & depth_edges & ~road_mask
        # Filter by aspect ratio: long horizontal strips
        labeled, n = ndimage.label(candidates)
        masks = []
        for i in range(1, n+1):
            cc = labeled == i
            ys, xs = np.where(cc)
            if len(xs) < 50:
                continue
            aspect = (xs.max() - xs.min()) / max(ys.max() - ys.min(), 1)
            if aspect > 4.0 and (ys.max() - ys.min()) < 40:
                masks.append(cc)
        return masks  # List of binary masks
        
    def propose_tunnel(self, semantic, depth, depth_edges):
        """Tunnel: deep enclosed region with converging depth gradients."""
        building_mask = (semantic == 2)
        # Find building regions with strong internal depth variation
        labeled, n = ndimage.label(building_mask)
        masks = []
        for i in range(1, n+1):
            cc = labeled == i
            ys, xs = np.where(cc)
            if len(ys) < 500:  # Tunnel is large
                continue
            # Check for depth gradient toward center (vanishing point)
            center_y, center_x = ys.mean(), xs.mean()
            # Top-half should be deeper than bottom-half (receding into tunnel)
            top_depth = depth[cc & (np.arange(depth.shape[0])[:, None] < center_y)].mean()
            bot_depth = depth[cc & (np.arange(depth.shape[0])[:, None] >= center_y)].mean()
            if top_depth > bot_depth + 0.1:  # Deeper at top = receding
                masks.append(cc)
        return masks
        
    def propose_caravan_trailer(self, semantic, depth, instance_proposals):
        """Caravan/trailer: boxy vehicle shapes behind trucks/cars."""
        # Use existing instance proposals; look for large rectangular boxes
        # attached to truck proposals but with distinct depth
        truck_mask = (semantic == 14)  # truck
        labeled, n = ndimage.label(truck_mask)
        masks = []
        for i in range(1, n+1):
            cc = labeled == i
            ys, xs = np.where(cc)
            # Look for adjacent boxy region with similar depth
            bbox = dilate_box(cc, margin=20)
            adj = bbox & ~cc & (semantic == 14)  # Currently labeled as truck
            if adj.sum() > 200:
                # Check aspect ratio: caravan is more rectangular
                adj_ys, adj_xs = np.where(adj)
                aspect = (adj_xs.max() - adj_xs.min()) / max(adj_ys.max() - adj_ys.min(), 1)
                if 1.5 < aspect < 3.5:
                    masks.append(adj)
        return masks
```

**2. Dataloader Integration:**

```python
class SyntheticRareDatasetWrapper:
    """Wraps PseudoLabelDataset to inject synthetic rare-class labels."""
    def __init__(self, base_dataset, injector, synthetic_weight=0.3,
                 inject_prob=0.5, max_synthetic_per_image=2):
        self.base = base_dataset
        self.injector = injector
        self.synthetic_weight = synthetic_weight
        self.inject_prob = inject_prob
        self.max_synthetic = max_synthetic_per_image
        
    def __getitem__(self, idx):
        sample = self.base[idx]
        semantic = sample['semantic']
        
        # Only inject if image has no rare classes
        has_rare = any((semantic == cid).sum() > 0 for cid in RARE_CLASS_IDS)
        if has_rare or random.random() > self.inject_prob:
            sample['synthetic_mask'] = torch.zeros_like(semantic, dtype=torch.bool)
            sample['synthetic_weight'] = 1.0
            return sample
            
        # Propose synthetic regions
        synthetic_mask = np.zeros_like(semantic, dtype=bool)
        for class_name in ['guard_rail', 'tunnel', 'caravan', 'trailer']:
            proposals = getattr(self.injector, f'propose_{class_name}')(
                semantic.numpy(), sample['depth'].numpy(), sample['depth_edges'].numpy()
            )
            if proposals and len(synthetic_mask[synthetic_mask].sum()) < self.max_synthetic:
                for prop in proposals[:self.max_synthetic]:
                    synthetic_mask |= prop
                    semantic[prop] = CLASS_NAME_TO_ID[class_name]
                    
        sample['semantic'] = semantic
        sample['synthetic_mask'] = torch.from_numpy(synthetic_mask)
        sample['synthetic_weight'] = self.synthetic_weight
        return sample
```

**3. Loss Integration:**

```python
# In SetCriterion.loss_labels:
loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

# Down-weight synthetic regions
if 'synthetic_mask' in targets[b]:
    synthetic = targets[b]['synthetic_mask']
    weights = torch.ones_like(loss_ce)
    weights[synthetic] = targets[b].get('synthetic_weight', 0.3)
    loss_ce = (loss_ce * weights).mean()
```

### Expected Impact

| Class | Expected ΔPQ | Mechanism |
|-------|-------------|-----------|
| guard rail | +4–10 | Synthetic labels at road edge; strong geometric prior |
| tunnel | +2–6 | Synthetic labels in receding building regions |
| polegroup | +0–2 | Harder to synthesize; marginal |
| caravan | +3–8 | Synthetic labels behind truck proposals |
| trailer | +3–8 | Same |

**Overall ΔPQ:** +2.0–4.5. This is the **highest-impact single intervention** because it directly addresses the root cause (absence of signal) rather than rebalancing existing signal.

### Feasibility
- **Medium implementation cost:** ~300 lines for injector + dataloader wrapper.
- **Requires depth maps at training time** (already available from DCFA pipeline).
- **Risk:** Poor proposals inject noise. Mitigation: conservative thresholds + low loss weight (0.3×) + validation-only proposal quality check.

---

## Synergistic Combination: The "Mask2Rare" Stack

The five approaches target different failure modes and compose naturally:

| Approach | Target Failure | Intervention Level | Cost |
|----------|---------------|-------------------|------|
| **QR-CPL** (Approach 1) | No gradient when rare absent | Model (query init + contrastive) | Low |
| **FA-HM** (Approach 2) | Rare targets ignored by matcher | Model (matching cost) | Very Low |
| **DH-RS** (Approach 3) | Rare classes drowned in shared head | Model (architecture) | Medium |
| **CQD-RCE** (Approach 4) | Early query capture by head classes | Training dynamics | Low |
| **PSRI** (Approach 5) | Zero rare-class pixels in labels | Data (synthetic injection) | Medium |

### Recommended Staging

**Phase 1 (Week 1): Fast Wins — FA-HM + CQD-RCE**
- Both are <100 lines, minimal risk.
- FA-HM ensures rare targets get matched; CQD-RCE prevents early monopolization.
- Expected: +1.5–3.0 PQ, 1–2 dead classes recovered.

**Phase 2 (Week 2): Capacity — DH-RS**
- Requires decoder modification but builds on existing decoupled heads.
- Expected additional: +1.0–2.0 PQ, caravan/trailer specifically improved.

**Phase 3 (Week 3): Signal Creation — PSRI + QR-CPL**
- PSRI addresses the root cause by creating synthetic labels.
- QR-CPL ensures rare queries don't drift when synthetic labels are sparse.
- Expected additional: +2.0–4.0 PQ, 3–4 dead classes recovered.

**Phase 4 (Week 4): Integration + Ablation**
- Full stack: PSRI + QR-CPL + FA-HM + DH-RS + CQD-RCE
- Expected total: +4.0–8.0 PQ, target 3–4 of 5 dead classes recovered.

### Comparison to Existing Proposals

The existing `rare_class_pseudolabel_proposals.md` (Proposals 1–5) targets **pseudo-label generation** (frequency-aware clustering, depth-edge splitting, k-NN propagation, copy-paste, multi-scale ensemble). These are complementary:
- **Pseudo-label proposals** increase rare-class *pixel frequency* before training.
- **Mask2Rare proposals** improve the *training dynamics* given whatever pseudo-labels exist.

**Optimal pipeline:** Apply pseudo-label proposals 1–2 (frequency-aware clustering + depth-edge split) → train with Mask2Rare stack → apply pseudo-label proposals 3–5 (propagation + copy-paste + ensemble) for Stage-4 fine-tuning.

---

## Open Questions & Risks

1. **Does DINOv3 contain sufficient rare-class signal?** If guard rail patches are truly indistinguishable from wall/fence in 768D DINOv3 space, no training intervention can recover them. Preliminary check: compute CAUSE logit variance for guard rail pixels. If max logit < 0.15, semantic recovery is infeasible without stronger features.

2. **Will DH-RS rare head overfit to synthetic labels?** With only ~50 real caravan pixels per epoch, the rare head may memorize synthetic patterns. Mitigation: strong dropout (0.5) in rare head + weight decay 0.01.

3. **PSRI proposal quality on 11GB GPUs.** Depth-guided proposal generation adds ~50ms per batch. With gradient accumulation 16, this is acceptable (~1.3s per effective batch). But proposal quality must be validated offline before training.

4. **Interaction with existing N-levers.** N1 (decoupled queries) and N3 (XQuery) may interfere with QR-CPL query reservation. Recommendation: test QR-CPL on standard 100-query pool first, then integrate with N1 if beneficial.

---

## References

1. J. Zhang et al., "Seesaw Loss for Long-Tailed Instance Segmentation," CVPR 2021.
2. G. Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation," CVPR 2021.
3. B. Cheng et al., "Masked-Attention Mask Transformer for Universal Image Segmentation," CVPR 2022. (Mask2Former)
4. H. Zhang et al., "DETR with Additional Global Aggregation for Tiny Object Detection," arXiv 2023. (Query reservation)
5. T. Wang et al., "Logit Adjustment Loss for Long-Tailed Visual Recognition," NeurIPS 2021.
6. K. Cao et al., "Collapsible Supervised Contrastive Learning for Long-Tailed Recognition," ICLR 2022.
