# Appendix Section A — Narrative Quality Audit

Scope: lines 441–481 of `paper/mbps_neurips2026_revised_compressed.tex`. Three subsections, three figures, one rendering-protocol paragraph.

## 1. Per-line audit

| Line(s) | Offending text (verbatim) | Issue category | Fix proposal |
|---|---|---|---|
| 448 | `Figure~\ref{fig:supp_real_data_flow} traces the Stage-1 data flow on real Cityscapes images, showing intermediate pseudo-label outputs together with the paired RGB/label transformations used by panoptic bootstrapping and panoptic network training.` | AI-slop ("showing intermediate", redundant "real Cityscapes", "paired RGB/label transformations" jargon-without-payload), forbidden colon-equivalent comma chains | Rewrite as a single statement of what the figure traces, anchored to the Stage-1 generator $G$ from Section 3. |
| 453 | `Columns: RGB; DepthPro depth; CAUSE-TR~\cite{kim2024cause} raw $K{=}80$ clusters; DCFA-adjusted $K{=}80$; depth-CC instances on RGB; final SIMCF-filtered panoptic pseudo-labels on RGB.` | Forbidden colon in caption prose; multi-line caption (rule 4) | Collapse into a one-line caption listing only the panel order with commas, no colon. |
| 453 | `Row 2 (hamburg) is the canonical co-planar pedestrian failure case discussed in the main paper's Conclusion section.` | Weakness language ("failure case"), back-pointer to Conclusion that signals an apologetic note (rule 5), narration that the figure should speak for itself (rule 4) | Delete entirely; the figure stands on its own without the failure-case label. |
| 457 | `\subsection{Cross-Dataset Qualitative Behavior}` followed immediately by line 460 `The cross-dataset transfer numbers in Table~\ref{tab:transfer}` | Disconnected from the previous subsection (rule 1) — Stage-1 trace ends on a pseudo-label flow and the next sentence jumps to cross-dataset numbers without a connector | Add a one-sentence bridge that carries the Stage-1 perspective forward into cross-dataset evaluation. |
| 460 | `which renders three held-out images from each of Cityscapes, KITTI, Mapillary Vistas v2 and Waymo V2 through the same Hungarian alignment that the PQ metric uses` | AI-slop scaffolding ("which renders ... through the same X that Y uses") | Tighten to a direct statement that the alignment used for evaluation is also the alignment used for rendering. |
| 460 | `Stuff and thing colors are therefore comparable across rows, and the trained network is evaluated on every image without retraining or per-dataset adaptation.` | Over-transparent / defensive ("without retraining or per-dataset adaptation") — signals what was *not* done (rule 5) | Drop "without retraining or per-dataset adaptation"; the cross-dataset table already establishes this. |
| 465 | `MBPS panoptic predictions on twelve held-out images, three per dataset, rendered under the Cityscapes-27 Hungarian alignment.` | Acceptable but uses comma-clause stack | Compress to one clean line (see §2). |
| 469 | `The same protocol governs the side-by-side comparison against the CUPS~\cite{hahn2025cups} baseline shown in Figure~\ref{fig:supp_ablation_viz}, where one image per dataset is processed by both the released CUPS checkpoint and our final Stage-3 model under identical inference settings.` | AI-slop ("The same protocol governs ... where ... is processed by both"), defensive over-specification ("under identical inference settings") | Replace with a tighter motivating sentence that names the comparison and the figure in one breath. |
| 474 | `Side-by-side panoptic predictions of CUPS~\cite{hahn2025cups} and MBPS on one held-out image per dataset.` | Acceptable; minor over-specification | Compress to one clean line (see §2). |
| 478 | `\subsection{Rendering Protocol}` followed by line 481 starting `All overlay panels in this appendix follow the same rendering protocol, applied symmetrically to MBPS and to the CUPS baseline so the comparison stays consistent.` | Disconnected from the previous figure paragraph (rule 1); "applied symmetrically ... so the comparison stays consistent" is defensive and AI-slop (rule 5) | Add a connector that picks up the CUPS-vs-MBPS thread from the prior subsection, then state the protocol without the "stays consistent" justification. |
| 481 | `Raw pseudo-cluster IDs are first projected to the 27-class Cityscapes color palette through the Hungarian assignment used by the PQ metric.` | Forbidden colon-style construction acceptable, but "first projected ... through the Hungarian assignment used by the PQ metric" repeats the Hungarian-alignment statement from line 460 — redundant across paragraphs (rule 1) | Restate once, in this paragraph, and remove the duplicate from line 460. |
| 481 | `Each thing instance is then assigned a colour from a shuffled \texttt{tab20} categorical palette and blended at 45\% with its semantic-class colour, so that adjacent instances of the same class remain visually distinct while preserving class identity.` | "so that ... remain visually distinct while preserving class identity" is an AI-slop justification clause (rule 2) | Drop the "so that ..." clause; the protocol description is sufficient on its own. |
| 481 | `The protocol is purely a rendering choice and does not modify the underlying panoptic predictions.` | Defensive / over-transparent — signals reviewer worry (rule 5) | Delete. |

## 2. Per-figure caption rewrite

### `fig:supp_real_data_flow` (line 453)

```latex
\caption{Stage-1 pseudo-label trace on four Cityscapes train frames with panels ordered RGB, DepthPro depth, raw $K{=}80$ clusters, DCFA-adjusted $K{=}80$, depth-CC instances overlay, and SIMCF-filtered panoptic overlay.}
```

### `fig:supp_cross_dataset` (line 465)

```latex
\caption{MBPS panoptic predictions on three held-out images from each of Cityscapes, KITTI, Mapillary Vistas v2, and Waymo V2 rendered under the Cityscapes-27 alignment.}
```

### `fig:supp_ablation_viz` (line 474)

```latex
\caption{CUPS and MBPS panoptic predictions on one held-out image from each of Cityscapes, KITTI, Mapillary Vistas v2, and Waymo V2.}
```

All three captions are one line, contain no `:`, no `--`, no `\textbf{}`, and no narrating verbs.

## 3. Paragraph-flow audit

**A.1 Stage-1 Pseudo-Label Trace.** This is the opening subsection of the appendix. Its first sentence (line 448) follows the appendix header directly and so needs no inter-subsection connector, but it does need a bridge from the main paper. The current opener jumps straight into a figure reference. Proposed connector: open with a sentence that carries the Stage-1 generator $G$ from Section 3 into the appendix viewing.

**A.2 Cross-Dataset Qualitative Behavior.** First sentence (line 460) jumps from a Stage-1 pseudo-label flow to a transfer table. The two are not connected. Proposed connector to insert at the top of A.2: a one-sentence pivot that moves from the Stage-1 generation perspective shown in A.1 to the trained-network behavior on held-out domains.

**A.3 Rendering Protocol.** First sentence (line 481) starts with "All overlay panels in this appendix follow the same rendering protocol", which is a topic shift away from the CUPS-vs-MBPS comparison that ends A.2. Proposed connector: open A.3 by referring to the comparison just shown and stating that the comparability of those panels depends on a single rendering convention, then describe it.

## 4. Replacement subsection drafts

### A.1 Stage-1 Pseudo-Label Trace

```latex
\subsection{Stage-1 Pseudo-Label Trace}
\label{sec:supp:stage1_trace}

The Stage-1 generator $G$ defined in Section~\ref{sec:method:baseline} composes a frozen semantic prior, a depth-conditioned residual, and a depth-cued instance proposal stream into the panoptic pseudo-label that supervises Stage-2. Figure~\ref{fig:supp_real_data_flow} renders this composition on four Cityscapes train frames, taking each image through every intermediate output that $G$ produces.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\textwidth]{../figures/paper_ready/fig5_stage1_trace_composite.png}
\caption{Stage-1 pseudo-label trace on four Cityscapes train frames with panels ordered RGB, DepthPro depth, raw $K{=}80$ clusters, DCFA-adjusted $K{=}80$, depth-CC instances overlay, and SIMCF-filtered panoptic overlay.}
\label{fig:supp_real_data_flow}
\end{figure}
```

### A.2 Cross-Dataset Qualitative Behavior

```latex
\subsection{Cross-Dataset Qualitative Behavior}
\label{sec:supp:cross_dataset_qualitative}

Once Stage-1 supervision is in place, the trained Stage-3 network is evaluated on held-out driving domains under the protocol of Section~\ref{sec:exp:generalization}. Figure~\ref{fig:supp_cross_dataset} extends the quantitative transfer numbers in Table~\ref{tab:transfer} into the image domain, with three frames drawn from Cityscapes, KITTI, Mapillary Vistas v2, and Waymo V2. Stuff and thing colors are produced under the Cityscapes-27 Hungarian alignment that the PQ metric uses, so each row is comparable to every other row.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.92\linewidth]{../figures/paper_ready/supp_qualitative/option_1_cross_dataset_qualitative.png}
\caption{MBPS panoptic predictions on three held-out images from each of Cityscapes, KITTI, Mapillary Vistas v2, and Waymo V2 rendered under the Cityscapes-27 alignment.}
\label{fig:supp_cross_dataset}
\end{figure}

The same alignment supports a like-for-like comparison against the CUPS~\cite{hahn2025cups} baseline. Figure~\ref{fig:supp_ablation_viz} pairs the released CUPS checkpoint with our Stage-3 model on one frame from each of the four domains.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.92\linewidth]{../figures/paper_ready/supp_qualitative/option_2_ablation_viz.png}
\caption{CUPS and MBPS panoptic predictions on one held-out image from each of Cityscapes, KITTI, Mapillary Vistas v2, and Waymo V2.}
\label{fig:supp_ablation_viz}
\end{figure}
```

### A.3 Rendering Protocol

```latex
\subsection{Rendering Protocol}
\label{sec:supp:visualization}

The CUPS-vs-MBPS panels above and every other overlay in this appendix share a single rendering convention. Raw pseudo-cluster IDs are projected to the 27-class Cityscapes color palette through the same Hungarian assignment that the PQ metric applies. Each thing instance then receives a color drawn from a shuffled \texttt{tab20} categorical palette, blended at 45\% with its semantic-class color.
```

## 5. Single one-paragraph summary

The three changes that will most improve narrative quality are the following. First, every figure caption must collapse to one line and must drop the colon, the semicolon-separated panel list, and the back-pointer to the Conclusion that currently appears in `fig:supp_real_data_flow`. Second, the appendix must read as a continuous arc starting from the Stage-1 generator and ending at the rendering convention that ties every panel together; this requires the three connectors proposed in §3 (Stage-1 generator opener, trained-network pivot at A.2, and CUPS-comparison handoff at A.3). Third, every clause that signals what was *not* done or that defends a choice must be deleted, including "without retraining or per-dataset adaptation", "stays consistent", "purely a rendering choice and does not modify the underlying panoptic predictions", and the failure-case mention of row 2 in Figure A.1.
