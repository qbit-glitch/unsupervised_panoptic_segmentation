# Appendix B and C Narrative Audit
**File**: `paper/mbps_neurips2026_revised_compressed.tex` (lines 484-609)
**Scope**: Appendix Section B "Implementation Details" and Section C "SIMCF Step Ablation and Threshold Sensitivity"
**Date**: 2026-05-03

---

## 1. Per-Line Audit Table

| Lines | Quoted text | Issue category | Fix proposal |
|-------|-------------|----------------|--------------|
| 488 | "expands the abbreviated experimental setup of the main paper" | AI-slop / weakness signal ("abbreviated" implies the main paper is incomplete) | Replace with "expands the experimental setup of the main paper" or "extends Section~\ref{sec:exp:setup} with the full hyperparameter, optimizer, augmentation, and hardware configuration." |
| 488 | "and were not retuned afterwards" | Apologetic / over-transparent (defensive against reviewer concern) | Drop or fold into a confident statement: "All operating hyperparameters were fixed before consulting Cityscapes annotations." |
| 493 | "(2{,}975 images, no annotations)" | Awkward parenthetical for a setup paragraph; reads like a side note rather than declarative prose | Rephrase: "use only the 2{,}975 unlabeled Cityscapes train images~\cite{cordts2016cityscapes}." |
| 499 | "DINOv2 ViT-B/14~\cite{oquab2024dinov2} backbone (no fine-tuning) and CAUSE-TR head~\cite{kim2024cause} (frozen, 90-dimensional concept-clusterbook code)." | Sentence fragment; not a complete sentence; reads like a bullet list | Promote to a sentence: "We freeze the DINOv2 ViT-B/14 backbone~\cite{oquab2024dinov2} and the CAUSE-TR head~\cite{kim2024cause}, which produces a 90-dimensional concept-clusterbook code." |
| 502 | "Loss: depth-correlation term $+\,\lambda_p \mathcal{L}_\text{preserve}$" | Forbidden colon in running prose | Replace with "The loss combines a depth-correlation term and $\lambda_p \mathcal{L}_\text{preserve}$ with $\lambda_p{=}20$." |
| 502 | "Optimizer: AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, 10K iterations, batch size 32 images, sampling 1024 pixels and 1024 pixel pairs per image." | Forbidden colon; comma-spliced bullet list inside prose | Convert to a proper sentence: "We optimize with AdamW (learning rate $10^{-3}$, weight decay $10^{-4}$) for 10K iterations at batch size 32, sampling 1024 pixels and 1024 pixel pairs per image." |
| 502 | "Wall-clock: $\approx 1$ hour on a single GTX 1080 Ti." | Forbidden colon | "DCFA training completes in roughly one hour on a single GTX 1080 Ti." |
| 504-505 | "$k{=}80$ centroids on L2-normalised DCFA-adjusted CAUSE-TR codes, fitted on the 2{,}975 Cityscapes train images using \texttt{scikit-learn} mini-batch $k$-means (batch 4096, 100 iterations, fixed random seed)." | Sentence fragment opening with a math symbol | Lead with a verb: "We fit $k{=}80$ centroids on L2-normalised DCFA-adjusted CAUSE-TR codes over the 2{,}975 Cityscapes train images using \texttt{scikit-learn} mini-batch $k$-means (batch 4096, 100 iterations, fixed random seed)." |
| 508 | "DepthPro~\cite{bochkovskii2024depthpro} monocular depth (frozen, no fine-tuning). $3\times3$ Sobel kernels, gradient threshold $\tau_d{=}0.20$, 8-neighbour connected components inside thing-eligible semantic regions, $A_\text{min}{=}1000$ pixel filter, three iterations of $3\times3$ dilation." | Two sentence fragments back-to-back; reads as a pasted parameter dump | Rewrite as: "Frozen monocular depth from DepthPro~\cite{bochkovskii2024depthpro} is filtered with $3\times3$ Sobel kernels at gradient threshold $\tau_d{=}0.20$. Inside thing-eligible semantic regions we extract 8-neighbour connected components, drop components below $A_\text{min}{=}1000$ pixels, and apply three iterations of $3\times3$ dilation." |
| 511 | "Feature similarity threshold $\tau_\text{sim}{=}0.85$, depth-implausibility multiplier $\eta{=}2.5$. Adjacent-proposal merging (Step~B) and depth-implausibility void rejection (Step~C) are applied in this order." | Fragment; bullet-style enumeration | "SIMCF uses feature similarity threshold $\tau_\text{sim}{=}0.85$ and depth-implausibility multiplier $\eta{=}2.5$, applying adjacent-proposal merging (Step~B) before depth-implausibility void rejection (Step~C)." |
| 521 | "Same augmentation pipeline as Stage-2; same optimizer family and learning-rate base ($5\times10^{-5}$ at the start of each round, cosine schedule per round)." | "Same X; same Y" reads as note-form rather than prose | Rewrite: "We reuse the Stage-2 augmentation pipeline and optimizer family with a base learning rate of $5\times10^{-5}$ at the start of each round under a per-round cosine schedule." |
| 526 | "consumer-grade hardware that lacks the calibrated stereo rigs and synchronized video required by CUPS" | Borderline weakness/comparison framing in Implementation Details (where it reads as defensive) | Keep but reframe positively: "the pipeline therefore runs end-to-end on consumer-grade hardware, without the calibrated stereo rigs or synchronized video that CUPS~\cite{hahn2025cups} requires at pseudo-label time." |
| 531 | "We make this explicit because reviewers may, on a quick reading, conflate the ``27'' in CAUSE's concept vocabulary with the ``27'' in the Cityscapes evaluation taxonomy" | AI-slop / over-transparent / addresses reviewer worry directly | Replace: "The two ``27''s in this paper refer to distinct objects: CAUSE's 27 self-supervised concept prototypes and the 27-class Cityscapes evaluation taxonomy. We disambiguate them here." |
| 531 | "these are distinct objects" preceded by ";" | Forbidden colon-style marker is fine; semicolon is OK, but the construction reads conversational | Same fix as above. |
| 540 | "The values of these hyperparameters were fixed before consulting any Cityscapes annotation." | Repeats line 488 verbatim in spirit | Drop here to avoid double-protest, or move only one of the two occurrences. |
| 555 | "Implementation: Python 3.10, PyTorch 2.1, Detectron2 0.6, CUDA 11.8." | Forbidden colon in prose | "We implement the pipeline in Python 3.10 with PyTorch 2.1, Detectron2 0.6, and CUDA 11.8." |
| 555 | "Stage-1 pseudo-labels and Stage-2/3 checkpoints will be released alongside the code upon publication." | Future-work / what-will-be-done language (mild) | Soften to factual: "Stage-1 pseudo-labels and Stage-2/3 checkpoints accompany the code release." |
| 562 | "We answer two questions in this appendix: how much each of A, B, C contributes in isolation, and whether the chosen thresholds are tuned-on-test." | Forbidden colon; "tuned-on-test" is reviewer-defensive phrasing | Rewrite: "This appendix isolates the per-step contribution of A, B, and C and characterises the sensitivity of the two thresholds." |
| 565 | "all variants sharing the DCFA-conditioned $K{=}80$ semantic source and DepthPro $\tau_d{=}0.20$ instance proposals." | Long trailing dependent clause; flows acceptably but the comma after $N{=}2975$ stacks two parentheticals | Tighten: "All variants share the DCFA-conditioned $K{=}80$ semantic source and DepthPro $\tau_d{=}0.20$ instance proposals." |
| 568 (caption) | Three-sentence caption that argues mechanism inside the caption | Caption rule: one line | See section 2 below for rewrite. |
| 580 | `\textbf{25.85}` and `\textbf{$+$0.63}` and `\textbf{14.70}` | Bold in a table is allowed by the rule, so this is fine; flag only that the prose claim must be motivated above | No fix needed; verify the prose precedes the table. |
| 585 | "the quantitative confirmation of the depth over-fragmentation diagnosis in Section~\ref{sec:method}." | Slightly self-congratulatory phrasing ("quantitative confirmation"); also a long clause | Tighten: "These statistics confirm the depth over-fragmentation behaviour described in Section~\ref{sec:method}." |
| 585 | "Pseudo-label PQ is a lower bound on Stage-3 trained PQ, so the $+0.63$ pseudo-label gain is consistent with the larger $+3.07$ PQ contribution-isolation reported at the trained level in Table~\ref{tab:main}." | Reasonable claim; "contribution-isolation" is jargon and the sentence is long | Split: "Because pseudo-label PQ lower-bounds Stage-3 trained PQ, the $+0.63$ pseudo-label gain is consistent with the $+3.07$ trained-PQ contribution reported in Table~\ref{tab:main}." |
| 588 | "To verify that SIMCF's contribution is not the product of threshold tuning, we sweep both thresholds $\pm{\sim}12\%$ around the paper setting" | "$\pm{\sim}12\%$" mixes two operators awkwardly; "is not the product of threshold tuning" is a defensive negation | Cleaner: "We probe sensitivity by sweeping both thresholds within roughly $\pm 12\%$ of the paper setting and re-evaluating Stage-1 pseudo-label quality." |
| 591 (caption) | Caption is two lines and contains parenthetical that argues distinctness from main-paper tables | Caption rule: one line, says what the table shows | See section 2 below for rewrite. |
| 591 | "(distinct from Cityscapes \textbf{val} numbers in main paper Tables~\ref{tab:components}/\ref{tab:conditioning})." | Defensive parenthetical; bold inside caption acceptable but the framing is reviewer-facing | Move the distinction into the prose, not the caption. |
| 607-608 | "\paragraph{Interpretation.}" + "Pseudo-label PQ varies within a 0.71-point band ... rather than tuned-on-test." | Section title "Interpretation" is fine, but "rather than tuned-on-test" is reviewer-defensive | Reframe positively: "The mechanism is therefore robust to moderate threshold perturbation." |

---

## 2. Per-Table Caption Rewrite (One Line Each)

**Table~\ref{tab:simcf_step_ablation}** (line 568):

```latex
\caption{Step-by-step SIMCF contribution to pseudo-label PQ on Cityscapes train under the 19-class many-to-one cluster-to-class mapping.}
```

**Table~\ref{tab:simcf_sensitivity}** (line 591):

```latex
\caption{Stage-1 pseudo-label quality on Cityscapes train as $\tau_\text{sim}$ and $\eta$ are swept around the paper setting.}
```

(The "distinct from main-paper Tables" disambiguation belongs in the surrounding prose, not the caption.)

---

## 3. Paragraph-Flow Audit per Subsection

**B opening (line 488) into B.1 Datasets (line 490)**: The opener promises "the full hyperparameter, optimizer, augmentation, and hardware configuration." The first thing the reader meets is datasets and metric protocol, not hyperparameters. Either reorder so Stage-1 hyperparameters land first, or revise the opener to "data, evaluation protocol, hyperparameters, optimizer, augmentation, and hardware." Recommended: revise opener.

**B.1 Datasets (493) into B.2 Stage-1 (495)**: The B.1 last sentence ends on cross-dataset transfer protocols. B.2 opens with "Frozen priors." There is no bridge from "evaluation on val" to "Stage-1 pseudo-label generation." Add a single connector sentence at the start of B.2: "We now describe Stage-1 pseudo-label generation, beginning with the frozen priors." Or merge the heading into a flowing paragraph.

**B.2 Stage-1 paragraphs (498-511)**: Each `\paragraph{}` block is essentially a fragment list. The four sub-paragraphs have no connective tissue between them. The reader jumps from "frozen priors" to "DCFA training" to "$k$-means" to "instance branch" to "SIMCF." Add explicit ordering cues: "Given these priors, we train DCFA," then "On the DCFA-adjusted codes we fit $k$-means," then "In parallel, the instance branch consumes monocular depth," then "SIMCF then reconciles the two branches."

**B.2 (511) into B.3 Stage-2 (513)**: B.2 ends on "Step C." B.3 opens with "Cascade Mask R-CNN." Add "Stage-2 takes the SIMCF-filtered pseudo-labels as supervision." or fold it into the first sentence of B.3.

**B.3 Stage-2 (516) into B.4 Stage-3 (518)**: Stage-2 ends on "wall-clock 12 hours." Stage-3 opens with "Three EMA self-training rounds." A connector such as "After Stage-2 converges we run three EMA self-training rounds" gives continuity.

**B.4 Stage-3 (521) into B.5 Hardware (523)**: Stage-3 already cites the GPU. B.5 then re-cites the same GPUs. Either drop the per-stage wall-clock from B.3/B.4 and keep them only in B.5, or rephrase B.5 as a roll-up: "Aggregating the per-stage budgets above..."

**B.5 (526) into B.6 $\phi$ map (528)**: Hardware ends on consumer-grade hardware. The next subsection opens by defending against a label-leakage misreading. The transition is the worst in the whole appendix. Add a connector that anchors $\phi$ in the pipeline rather than in reviewer worry. Suggested opener: "Both the instance branch and SIMCF rely on a fixed map $\phi : \{1, \dots, 80\} \to \{1, \dots, 27\}$ from over-cluster indices to CAUSE concepts. We describe how $\phi$ is constructed, and disambiguate the two ``27''s that appear in the paper."

**B.6 (540) into B.7 DCFA Loss (542)**: B.6 ends on a guarantee about hyperparameters. B.7 opens with notation. Add: "We now define the two DCFA loss terms referenced in Section~\ref{sec:method:dcfa}." or merge the heading.

**B.7 (550) into B.8 Reproducibility (552)**: Loss ends on bandwidth. Reproducibility opens on seeds. Add: "For reproducibility we fix the following seeds and software stack."

**C "Step ablation" (564-585) into "Threshold sensitivity" (587)**: This is the bridge the user explicitly flagged. Step-ablation paragraph closes with a comparison to Table~\ref{tab:main}. "Threshold sensitivity" then opens with "To verify that SIMCF's contribution is not the product of threshold tuning." The two paragraphs feel slap-together because (a) the second opens on a defensive negation, and (b) there is no explicit "Having shown that B carries the gain, we now ask whether the chosen thresholds are themselves the source of the gain." A one-sentence connector fixes this. See replacement draft in section 4.

**C "Threshold sensitivity" (587-605) into "Interpretation" (607)**: The interpretation paragraph paraphrases the table without first stating what the table established. Tighten the connector: "The sweep, summarised in Table~\ref{tab:simcf_sensitivity}, varies pseudo-label PQ within a 0.71-point band..." See section 4.

---

## 4. Replacement Subsection Drafts

### 4.1 Replacement for B.2 "Stage-1: Pseudo-Label Generation" (lines 495-511)

```latex
\subsection{Stage-1: Pseudo-Label Generation}
\label{sec:supp:impl:stage1}

Stage-1 produces semantic and instance pseudo-labels from frozen, self-supervised priors. We freeze the DINOv2 ViT-B/14 backbone~\cite{oquab2024dinov2} and the CAUSE-TR head~\cite{kim2024cause}, which together produce a 90-dimensional concept-clusterbook code per pixel.

DCFA is a two-hidden-layer MLP of width $h{=}384$ acting on the concatenated input $[z_u; e(d_u)] \in \mathbb{R}^{106}$, where $e(d_u)$ is a 16-D sinusoidal embedding of monocular depth, and contributes about 225K trainable parameters. The training objective combines a depth-correlation term and $\lambda_p \mathcal{L}_\text{preserve}$ with $\lambda_p{=}20$, anchoring the residual update to identity at initialisation. We optimize with AdamW (learning rate $10^{-3}$, weight decay $10^{-4}$) for 10K iterations at batch size 32, sampling 1024 pixels and 1024 pixel pairs per image. DCFA training completes in roughly one hour on a single GTX 1080 Ti.

We then fit $k{=}80$ centroids on L2-normalised DCFA-adjusted CAUSE-TR codes over the 2{,}975 Cityscapes train images using \texttt{scikit-learn} mini-batch $k$-means (batch 4096, 100 iterations, fixed random seed). The resulting centroids are saved once and reused throughout the rest of the pipeline.

In parallel, the instance branch consumes frozen monocular depth from DepthPro~\cite{bochkovskii2024depthpro}. We compute depth gradients with $3\times3$ Sobel kernels at threshold $\tau_d{=}0.20$, take 8-neighbour connected components inside thing-eligible semantic regions, drop components below $A_\text{min}{=}1000$ pixels, and apply three iterations of $3\times3$ dilation.

SIMCF then reconciles the two branches. It uses a feature similarity threshold $\tau_\text{sim}{=}0.85$ and a depth-implausibility multiplier $\eta{=}2.5$, applying adjacent-proposal merging (Step~B) before depth-implausibility void rejection (Step~C).
```

### 4.2 Replacement for B.6 "$\phi$ Map" (lines 528-540)

```latex
\subsection{The Over-Cluster-to-Concept Map $\phi$ Uses No Ground-Truth Labels}
\label{sec:supp:impl:phi}

Both the instance branch and SIMCF rely on a fixed map $\phi : \{1, \dots, 80\} \to \{1, \dots, 27\}$ from over-cluster indices to CAUSE concept prototypes. We document its construction here and disambiguate the two ``27''s that appear in the paper, since the same number is used for two unrelated objects: CAUSE's 27 self-supervised concept prototypes and the 27-class Cityscapes evaluation taxonomy.

\paragraph{What $\phi$ maps between.}
The source side is the set of 80 $k$-means cluster centroids fitted on the DCFA-adjusted 90-D semantic codes of Cityscapes train (Section~\ref{sec:method:baseline}). The target side is the 27 concept prototypes inside the frozen CAUSE-TR head~\cite{kim2024cause}. CAUSE-TR was trained without any human annotation, on Cityscapes train images using only the self-supervised concept-coherence objective, so its 27 prototypes form an internally-discovered partition of the latent space rather than a label space.

\paragraph{How $\phi$ is computed.}
For each $k \in \{1, \dots, 80\}$, $\phi(k) = \arg\min_{c \in \{1, \dots, 27\}} \|\mu_k - p_c\|_2$, where $\mu_k$ is the $k$-th $k$-means centroid and $p_c$ is the $c$-th CAUSE concept prototype, both in the 90-D code space. This is a deterministic nearest-centroid lookup that runs once at the end of $k$-means fitting. The thing/stuff partition over the 27 CAUSE concepts (8 thing categories, 19 stuff categories) is fixed by CAUSE-TR's pretraining and read off the frozen head.

\paragraph{Where ground-truth enters the pipeline.}
The only ground-truth-derived component is the Hungarian map between the 27 CAUSE concept prototypes and the 27 Cityscapes evaluation classes, computed once over the 500-image Cityscapes val set at metric time (Section~\ref{sec:exp:setup}). The Hungarian map is used only to compute PQ, SQ, and RQ, and never feeds back into the Stage-1 generator $G$, the panoptic network $F$, or any of the operating hyperparameters $(\tau_d, \tau_\text{sim}, \eta, A_\text{min}, \lambda_p, k)$.
```

### 4.3 Replacement for §C SIMCF Step Ablation (lines 562-608)

```latex
\section{SIMCF Step Ablation and Threshold Sensitivity}
\label{sec:supp:simcf_sweep}

SIMCF exposes two thresholds and three composable steps. The thresholds are $\tau_\text{sim}$ for adjacent-proposal feature merging in Step~B and $\eta$ for depth-implausibility void rejection in Step~C. The three steps are A (intra-instance majority vote), B (cosine-merge), and C (depth void). This appendix isolates the per-step contribution of A, B, and C, then characterises the sensitivity of the two thresholds.

\paragraph{Step ablation.}
Table~\ref{tab:simcf_step_ablation} reports pseudo-label PQ on Cityscapes train ($N{=}2975$) with steps incrementally enabled. All variants share the DCFA-conditioned $K{=}80$ semantic source and DepthPro $\tau_d{=}0.20$ instance proposals.

\begin{table}[!htbp]
\caption{Step-by-step SIMCF contribution to pseudo-label PQ on Cityscapes train under the 19-class many-to-one cluster-to-class mapping.}
\label{tab:simcf_step_ablation}
\centering
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{l c c c c c c}
\toprule
Setting & PQ & $\Delta$PQ & PQ$^\text{st}$ & PQ$^\text{th}$ & mIoU & Ignore (\%) \\
\midrule
DCFA only (no SIMCF)        & 25.22 & ---     & 33.99 & 13.16 & 56.16 & 0.00 \\
\, $+$ SIMCF-A              & 25.22 & $+$0.00 & 33.99 & 13.16 & 56.16 & 0.00 \\
\, $+$ SIMCF-A$+$B          & 25.84 & $+$0.62 & 33.99 & 14.64 & 56.16 & 0.00 \\
\, $+$ SIMCF-A$+$B$+$C      & \textbf{25.85} & \textbf{$+$0.63} & 33.96 & \textbf{14.70} & 56.22 & 0.84 \\
\bottomrule
\end{tabular}
\end{table}

Step~B carries the entire PQ$^\text{th}$ gain. The instance map is touched only by Step~B, where per-image instance count drops from 16.2 to 7.0 and median instance size rises from 6{,}372 to 16{,}413 pixels. These statistics confirm the depth over-fragmentation behaviour described in Section~\ref{sec:method}. Step~A is inert on DCFA-conditioned codes because within-instance cluster purity is already high, and Step~C trades 0.84\% void for marginal stuff cleanup. Because pseudo-label PQ lower-bounds Stage-3 trained PQ, the $+0.63$ pseudo-label gain is consistent with the $+3.07$ trained-PQ contribution reported in Table~\ref{tab:main}.

\paragraph{Threshold sensitivity.}
Having localised the gain to Step~B, we next ask whether the specific values $\tau_\text{sim}{=}0.85$ and $\eta{=}2.5$ are themselves load-bearing. We sweep both thresholds within roughly $\pm 12\%$ of the paper setting and re-evaluate Stage-1 pseudo-label quality on the full Cityscapes train set under the CUPS~\cite{hahn2025cups} 27-class protocol. The sweep is therefore directly comparable to the Stage-1 numbers in Table~\ref{tab:components} and not to the trained-PQ numbers elsewhere in the main paper.

\begin{table}[!htbp]
\caption{Stage-1 pseudo-label quality on Cityscapes train as $\tau_\text{sim}$ and $\eta$ are swept around the paper setting.}
\label{tab:simcf_sensitivity}
\centering
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{l c c c c c c}
\toprule
Setting & $\tau_\text{sim}$ & $\eta$ & PQ & SQ & RQ & mIoU \\
\midrule
tighter           & 0.75 & 2.0 & 25.98 & 58.79 & 34.55 & 56.08 \\
baseline (paper)  & 0.85 & 2.5 & 25.87 & 58.38 & 34.83 & 56.17 \\
looser            & 0.95 & 3.0 & 25.27 & 57.63 & 34.10 & 56.22 \\
\bottomrule
\end{tabular}
\end{table}

Pseudo-label PQ varies within a 0.71-point band (25.27 to 25.98) across the sweep, well inside the $+1.31$ PQ gap from the raw $K{=}80$ baseline (24.54) to Both (25.85) reported in main-paper Table~\ref{tab:components}. Tighter thresholds slightly improve PQ and looser thresholds degrade modestly, so the mechanism is robust to moderate threshold perturbation.
```

### 4.4 Replacement for B opening + B.1 + Stage-3 + Hardware (lines 488-526)

A unifying connector pass (only the spans that need it). Replace lines 488-493 with:

```latex
This appendix extends Section~\ref{sec:exp:setup} with the full data, evaluation, hyperparameter, optimizer, augmentation, and hardware configuration used for every reported result.

\subsection{Datasets and Evaluation Protocol}
\label{sec:supp:impl:data}

Pseudo-label generation and panoptic network training use only the 2{,}975 unlabeled Cityscapes train images~\cite{cordts2016cityscapes}. Evaluation is on Cityscapes val (500 images) under the 27-class CAUSE~\cite{kim2024cause} pseudo-vocabulary with a Hungarian map to the 27-class Cityscapes label space (\texttt{NUM\_CLASSES=27} in the official CUPS protocol~\cite{hahn2025cups}). The 27-class evaluation taxonomy includes the 19 standard semantic-segmentation classes plus parking, rail track, guard rail, bridge, tunnel, polegroup, caravan, and trailer. Cross-dataset transfer follows the CUPS~\cite{hahn2025cups} protocol on KITTI~\cite{geiger2012kitti}, Waymo V2~\cite{sun2020waymo}, and Mapillary Vistas v2~\cite{neuhold2017mapillary}.
```

Replace lines 518-526 (Stage-3 opener + Hardware opener) with:

```latex
\subsection{Stage-3: EMA Self-Training}
\label{sec:supp:impl:stage3}

After Stage-2 converges, we run three EMA self-training rounds of 5K optimizer steps each with EMA coefficient 0.999. A three-scale teacher (scales $\{0.75, 1.0, 1.25\}$) supplies class-aware confidence thresholds (foreground 0.4, background 0.3). We reuse the Stage-2 augmentation pipeline and optimizer family with a base learning rate of $5\times10^{-5}$ at the start of each round under a per-round cosine schedule. Stage-3 takes roughly 16 wall-clock hours on $2\times$ GTX 1080 Ti.

\subsection{Hardware and Wall-Clock Budget}
\label{sec:supp:impl:hardware}

Aggregating the per-stage budgets above, all training runs on $2\times$ NVIDIA GTX 1080 Ti GPUs (consumer-grade, 11 GB VRAM each) under PyTorch DistributedDataParallel. Stage-1 generation accounts for $\approx 2$ GPU-hours, Stage-2 bootstrapping for $\approx 24$ GPU-hours ($\approx 12$ wall-clock on 2 GPUs), and Stage-3 self-training for $\approx 32$ GPU-hours ($\approx 16$ wall-clock), giving an aggregate of $\approx 58$ GPU-hours. The pipeline therefore runs end-to-end on consumer-grade hardware, without the calibrated stereo rigs or synchronized video that CUPS~\cite{hahn2025cups} requires at pseudo-label time.
```

---

## 5. Top-Three Summary

The three changes that will most improve narrative quality are: first, eliminate every colon in running prose and convert the four `\paragraph{}`-block fragments inside B.2 into connected sentences so Stage-1 reads as one continuous procedure rather than a parameter dump; second, rebuild the bridge between the two paragraphs of §C by opening "Threshold sensitivity" with "Having localised the gain to Step~B, we next ask whether the specific values are themselves load-bearing," so the appendix reads as one continuous argument from per-step decomposition to per-threshold robustness; third, remove every defensive or reviewer-facing phrase ("not retuned afterwards," "reviewers may conflate," "not the product of threshold tuning," "rather than tuned-on-test") and reframe each as a positive declaration so the appendix matches the confidence of a NeurIPS-quality main paper.
