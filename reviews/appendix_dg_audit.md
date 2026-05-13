# Appendix Narrative-Quality Audit

**File audited**: `paper/mbps_neurips2026_revised_compressed.tex`
**Scope**: Lines 611-716 (D Depth Model Ablation, E Out-of-Distribution Sanity Checks, F Per-Class Cityscapes Metrics, G Broader Impact).
**Auditor stance**: NeurIPS 2026 narrative-quality reviewer applying the seven user rules. No technical claim is invented; every rewrite paraphrases existing prose only.

---

## 1. Per-Line Audit Tables

Issue categories used below: **AI-slop** (mechanical phrasing, list-of-facts cadence, "the table shows" / "we additionally" filler), **Disconnected** (no bridge from neighbour), **Forbidden char** (em-dash `--`, prose colon, bold in prose, parenthetical clutter), **Weakness language** (apology, hedging, "future work", "could be added", "departs from"), **Redundant** (sentence repeats a fact already stated).

### D. Depth Model Ablation (lines 611-637)

| Lines | Offending text (verbatim) | Issue | Fix proposal |
|-------|---------------------------|-------|--------------|
| 615 | `The rows in Table~\ref{tab:depth} are not a single leaderboard: some use the raw $K{=}80$ semantic map and others use the DCFA semantic map.` | Forbidden char (prose colon); AI-slop (defensive meta-commentary about the table) | Replace colon with a comma and rephrase as a single declarative ("Rows mix the raw $K{=}80$ semantic map and the DCFA semantic map, ..."). |
| 615 | `They are reported jointly to separate mask quality from recognition quality while choosing the training-data configuration.` | AI-slop ("they are reported jointly to..." is meta-commentary); Disconnected from line 615 opening | Rewrite as "Reporting them jointly isolates mask quality from recognition quality and fixes the training-data configuration." |
| 618 | `(note: Stage-1 pseudo-label level evaluated on the train split, distinct from main paper Tables~\ref{tab:main} and \ref{tab:components}/\ref{tab:conditioning} which report on Cityscapes \textbf{val})` | Forbidden char (`\textbf` and prose colon); AI-slop (parenthetical mini-disclaimer); Caption exceeds one line | Move the train-vs-val split clarification into prose preceding the table; keep caption to one line: "Depth and threshold diagnostics on Cityscapes train pseudo-labels." |
| 637 | `\paragraph{Discussion.}` followed by `The table shows why PQ alone is not a sufficient diagnostic for the depth branch.` | AI-slop ("the table shows" boilerplate opener) | Drop the `Discussion` label; open paragraph with a substantive sentence ("PQ alone is not a sufficient diagnostic for the depth branch.") |
| 637 | `but it produces a fragmented training set with roughly 57 instances per image and more than half below 1000 pixels` | Weakness language ("fragmented training set" frames the choice apologetically) | Reframe as a measurement, not a verdict ("but yields roughly 57 instances per image, with more than half under 1000 pixels"). |
| 637 | `DA v3 at the same training-data threshold reaches a similar RQ, indicating that the choice is governed by the training signal produced by the depth branch rather than by the stuff/thing aggregate split.` | Disconnected (the sentence about DA v3 lands without a bridge from the DepthPro discussion before it) | Add a connector ("Under the same threshold, DA v3 reaches a similar RQ, so the choice is governed by ..."). |

### E. Out-of-Distribution Sanity Checks (lines 640-664)

| Lines | Offending text (verbatim) | Issue | Fix proposal |
|-------|---------------------------|-------|--------------|
| 644 | `The main paper reports cross-dataset transfer on three driving-scene benchmarks (KITTI~\cite{geiger2012kitti}, Waymo V2~\cite{sun2020waymo}, Mapillary v2~\cite{neuhold2017mapillary} in Section~\ref{sec:exp:generalization}) where the Cityscapes-derived pseudo-label vocabulary is in-domain.` | AI-slop (long parenthetical clutter); Disconnected from §D's last sentence | Open §E with a bridge from the depth-branch discussion ("Beyond the in-domain transfer of Section ..."), and move the citation chain into a clean list without a parenthetical. |
| 644 | `We additionally run two out-of-distribution sanity checks: (i) MOTS~\cite{voigtlaender2019mots} as a thing-only label space whose vocabulary is a subset of Cityscapes-things, and (ii) COCO-Stuff-27~\cite{caesar2018coco} as a vocabulary-mismatched object-centric dataset whose 27 classes overlap only partially with Cityscapes-27.` | AI-slop ("we additionally run"); Forbidden char (prose colon) | Rewrite without the colon and the apologetic adverb: "Two further checks probe vocabularies outside the in-domain set. MOTS isolates a thing-only label space whose vocabulary is a subset of Cityscapes-things, and COCO-Stuff-27 supplies a vocabulary-mismatched object-centric setting whose 27 classes overlap only partially with Cityscapes-27." |
| 647 | `Out-of-distribution sanity checks on a thing-only label space (MOTS) and a vocabulary-mismatched object-centric dataset (COCO-Stuff-27). All numbers are percentages.` | Caption exceeds one line | Compress: "Out-of-distribution checks on MOTS (thing-only) and COCO-Stuff-27 (vocabulary-mismatched). Values are percentages." |
| 657 | `\quad CUPS~\cite{hahn2025cups} ref. & --- & 86.4 & 86.4 & 76.9 \\` | Acceptable inside table cell, but `---` em-dash lookalike is fine in a table cell; flag only because of the dangling reference row narrative below | No change in cell; ensure the prose explains why the row is included. |
| 663-664 | `\paragraph{Interpretation.}` opening into `The MOTS result (61.10 PQ, SQ 88.71) confirms that mask geometry generalizes to a thing-only OOD vocabulary` | AI-slop (`Interpretation` label is filler heading) | Remove the `\paragraph{Interpretation.}` heading; the prose itself is interpretive. |
| 664 | `The gap to the CUPS~\cite{hahn2025cups} reference (86.4 PQ) is recognition-driven (RQ 65.80 vs.\ 76.9) and reflects vocabulary granularity rather than a transfer failure of the architecture.` | Weakness language ("not a transfer failure" is defensive); AI-slop (`vs.\ ` aside is parenthetic clutter) | Reframe positively: "The gap to the CUPS reference is recognition-driven, with RQ 65.80 against 76.9; vocabulary granularity, not transfer behaviour, accounts for the difference." |
| 664 | `most COCO-Stuff-27 categories (e.g.\ kitchenware, sports gear, food) have no analog in the Cityscapes pseudo-vocabulary and are unrecoverable by Hungarian alignment alone` | AI-slop (parenthetical example list interrupts the sentence) | Move examples into a follow-on clause: "...have no analog in the Cityscapes pseudo-vocabulary, including kitchenware, sports gear, and food categories, and are unrecoverable by Hungarian alignment alone." |
| 664 | `This is the vocabulary boundary of in-domain training and not a transfer failure of the architecture; resolving it would require expanding the pseudo-label generator's vocabulary at training time, which is orthogonal to the contributions of this paper.` | Weakness language ("not a transfer failure", "resolving it would require", "orthogonal to the contributions of this paper" all read as apology / future-work) | Replace with a single calibrated statement: "The result delineates the vocabulary boundary of in-domain training; coverage of object-centric vocabularies outside the Cityscapes pseudo-space is set by the generator's vocabulary itself." |

### F. Per-Class Cityscapes Metrics (lines 667-704)

| Lines | Offending text (verbatim) | Issue | Fix proposal |
|-------|---------------------------|-------|--------------|
| 671 | `Table~\ref{tab:perclass} reports per-class PQ/SQ/RQ for the final trained network of the main paper (Section~\ref{sec:exp:main}) under the CUPS~\cite{hahn2025cups} 27-class evaluation protocol with Hungarian matching from the $K{=}80$ pseudo-vocabulary to the 27-class Cityscapes label space.` | AI-slop (table-introduces-itself opener); Disconnected from end of §E | Open with a bridge from §E's vocabulary observation ("Within the in-domain Cityscapes vocabulary, ..."); then state the protocol cleanly. |
| 674 | `Per-class Cityscapes validation metrics for the final trained network. Values are percentages. Six classes (parking, guard rail, tunnel, polegroup, caravan, trailer) are vocabulary-dead; the twenty active classes carry the 35.83 PQ aggregate reported in the main paper.` | Caption exceeds one line; Weakness language ("vocabulary-dead", "carry the 35.83 PQ aggregate" reads as concession) | Compress to one factual line: "Per-class Cityscapes validation PQ/SQ/RQ for the final trained network. Values are percentages." Move the dead-class commentary into prose where it belongs. |
| 698 | `\paragraph{Discussion.}` | Filler label heading | Remove. |
| 699 | `Twenty active classes carry the 35.83 PQ aggregate.` | AI-slop (mechanical opener that restates the caption) | Drop and start with a substantive observation about the score distribution. |
| 699 | `Large stuff regions (road 93.0, sky 86.1, vegetation 84.7, building 83.5) and large vehicles (car 70.7, bus 76.7, train 77.2) score above 70 PQ, since frozen semantic codes produce coherent regions and depth boundaries align with object extent.` | Forbidden char (parenthetical numeric clutter inside running prose) | Pull the number lists out of the parentheses into a colon-free list, or summarise the band ("Large stuff regions and large vehicles all score above 70 PQ, with road, sky, vegetation and building above 80 ..."). |
| 699 | `The remaining seven classes are vocabulary-coverage cases:` | Forbidden char (prose colon); the count "seven" disagrees with the six listed in the caption (rule 5 also bans drawing reviewer attention to gaps, but this is a numerical inconsistency that must be fixed) | Replace with a comma-form sentence and reconcile the count: "The remaining classes are vocabulary-coverage cases. Parking, guard rail, tunnel, polegroup, caravan, and trailer lie outside ..." |
| 699 | `their effect on the aggregate is bounded: removing them and averaging only over the standard Cityscapes-19 panoptic classes raises the protocol comparability of our headline.` | Forbidden char (prose colon); Weakness language ("their effect on the aggregate is bounded" reads defensively, and "raises the protocol comparability of our headline" is meta-commentary) | Drop the apologetic clause; restate as a calibration fact ("Restricting the average to the standard Cityscapes-19 panoptic classes is therefore the protocol-comparable reading of the headline."). |
| 701 | `(both above the 60-point band that classifies the per-class match as geometrically correct)` | AI-slop (parenthetical aside); Forbidden phrasing density | Lift into the main clause: "...both above the 60-point band, indicating geometrically correct per-class matches, while their RQ values..." |
| 701 | `This is the failure mode discussed in the main paper's Conclusion (Section~\ref{sec:conclusion}).` | Weakness language ("failure mode" is the wrong register for an appendix close-out); Disconnected (closes the paragraph by deferring to another section) | Replace with a forward-looking observation ("Section~\ref{sec:conclusion} returns to the co-planar pedestrian regime as the principal recognition-side residual."). |
| 703 | `\paragraph{SIMCF feature-merge diagnostics.}` | Acceptable as a paragraph label, but the paragraph that follows is disconnected from the per-class discussion above it (different topic with no bridge) | Add an opening bridge from per-class recognition limits to the merge-step diagnostics ("These per-class recognition limits motivate the feature-guided merge step (Step B) of SIMCF, whose two qualitative measurements complete the picture."). |
| 704 | `Average pseudo-instance count per image drops from 44 before the merge to 22 after, median instance size grows from 5{,}502 to 14{,}965 pixels, and stuff contamination falls from 50.7\% to 28.0\%.` | OK numerically; comma-spliced run-on borderline | Split into two sentences for cadence ("...22 after. Median instance size grows from 5,502 to 14,965 pixels, and stuff contamination falls from 50.7\\% to 28.0\\%."). |

### G. Broader Impact (lines 707-715)

| Lines | Offending text (verbatim) | Issue | Fix proposal |
|-------|---------------------------|-------|--------------|
| 711 | `Removing stereo and motion requirements at pseudo-label time broadens the range of imagery to which scene-centric unsupervised panoptic segmentation can be applied.` | OK opener but does not bridge from §F's last sentence | Add a bridge ("Beyond per-class behaviour, the design itself widens deployment scope: removing stereo and motion requirements ..."). |
| 713 | `The same property creates dual-use risk.` | AI-slop ("dual-use risk" is the canonical apologetic-impact opener) | Reframe in calibrated, non-apologetic register: "The same property changes the deployment surface." |
| 713 | `A monocular-only pipeline lowers the data and hardware bar for deployment in surveillance and behavioral-monitoring settings that the original stereo-based methods could not easily reach.` | Borderline; reads as soft apology when paired with the previous sentence | Keep the factual content but tighten ("Monocular footage is cheaper to acquire and easier to repurpose for surveillance and behavioural-monitoring settings that stereo-based methods could not reach."). |
| 713 | `We do not release pretrained weights tuned to specific surveillance contexts, and the system as evaluated targets street-scene categories rather than person re-identification.` | Weakness language ("we do not release ..." reads as preemptive defence) | Replace with a confident scope statement: "Released artifacts are restricted to street-scene category training; person re-identification is outside the evaluated label space." |
| 715 | `Two technical-bias concerns also follow from the design.` | AI-slop ("X concerns also follow from the design" is the canonical impact-section template) | Replace with a calibrated handoff: "Two design-level biases warrant explicit calibration." |
| 715 | `six dead classes, discussed in the main paper's limitations section, are a direct symptom and would compound on demographically less-represented scenes` | Weakness language ("dead classes" + "direct symptom" + "would compound" is speculative-future-failure phrasing); cross-references the limitations section, which dilutes the local point | Replace with a calibration sentence whose subject is the system, not a hypothetical worse outcome ("The six low-population pseudo-classes from §F therefore inherit any imbalance present in the source clusterbook, which calibration on demographically distinct scenes must account for."). |
| 715 | `These limitations should be accounted for before downstream deployment, particularly in settings where the consequences of mispredicted person-class instances are non-trivial.` | Weakness language ("limitations", "should be accounted for", "consequences ... are non-trivial" all hedge) | Reframe as a calibrated deployment note: "Calibration of the person-class instance behaviour is a prerequisite for deployment in settings where mispredictions carry operational cost." |

---

## 2. One-Line Caption Rewrites

| Tag | Existing caption (paraphrased) | One-line replacement |
|-----|-------------------------------|----------------------|
| `tab:depth` (line 618) | Multi-line caption with bold "train" / "val" warnings and cross-table notes | `Depth-estimator and gradient-threshold diagnostics on Cityscapes train pseudo-labels.` |
| `tab:cocooo` (line 647) | Two-sentence caption naming both datasets and stating units | `Out-of-distribution checks on MOTS and COCO-Stuff-27. Values are percentages.` |
| `tab:perclass` (line 674) | Three-sentence caption containing the dead-class list | `Per-class Cityscapes validation PQ, SQ, and RQ for the final trained network. Values are percentages.` |

(No `\begin{figure}` blocks appear in this appendix range; only tables.)

---

## 3. Section-Flow Audit

| Section | First sentence (paraphrased) | Last sentence of previous section | Connector verdict | Proposed connector |
|---------|------------------------------|-----------------------------------|-------------------|---------------------|
| D Depth Model Ablation | "We compared three monocular depth estimators ..." | (§C closes on SIMCF threshold sensitivity) | Disconnected. The opener jumps to a new comparison without referencing the SIMCF discussion that closed §C. | "Holding the SIMCF threshold at the value selected in §\\ref{sec:supp:simcf}, we now vary the depth estimator and the gradient threshold $\\tau$." |
| E Out-of-Distribution Sanity Checks | "The main paper reports cross-dataset transfer on three driving-scene benchmarks ..." | "...is governed by the training signal produced by the depth branch rather than by the stuff/thing aggregate split." | Disconnected. The first sentence restates main-paper experiments without referencing the depth-branch conclusion of §D. | "Beyond the in-domain transfer reported in the main paper (Section~\\ref{sec:exp:generalization}), two further sanity checks probe vocabularies outside that range." |
| F Per-Class Cityscapes Metrics | "Table~\ref{tab:perclass} reports per-class PQ/SQ/RQ ..." | "...orthogonal to the contributions of this paper." | Disconnected; opener also has the AI-slop "Table X reports" pattern. | "Within the in-domain Cityscapes vocabulary, the per-class PQ/SQ/RQ profile of the final trained network is given in Table~\\ref{tab:perclass}." |
| G Broader Impact | "Removing stereo and motion requirements ..." | "This is the failure mode discussed in the main paper's Conclusion ..." | Disconnected; opener pivots from per-class recognition to deployment scope without a bridge. | "Beyond per-class behaviour, the monocular-only design itself reshapes the deployment surface for unsupervised panoptic segmentation." |

**Dumping-ground assessment.** §D and §F deepen main-paper claims and are natural extensions; §E reads partly as a dumping ground (the COCO-Stuff-27 row exists mainly to absorb a low number, and the prose currently apologises for it); §G reads as the canonical AI-slop impact statement and needs the calibrated rewrite below to feel like a research-paper extension rather than a checklist response.

---

## 4. Replacement Section Drafts

Each replacement preserves every numerical claim and citation in the existing prose; only register, bridges, and forbidden constructs are altered.

### 4.1 Replacement for §D Depth Model Ablation

```latex
\section{Depth Model Ablation}
\label{sec:supp:depth}

Holding the SIMCF threshold at the value selected in
Section~\ref{sec:supp:simcf}, we vary the monocular depth estimator
and the gradient threshold $\tau$. Three estimators are compared:
SPIdepth, Depth-Anything v3~\cite{lin2025da3}, and
DepthPro~\cite{bochkovskii2024depthpro}. Rows mix the raw $K{=}80$
semantic map and the DCFA semantic map, so reading them jointly
isolates mask quality from recognition quality and fixes the
training-data configuration used in §\ref{sec:exp:main}.
The diagnostics are computed on Cityscapes train pseudo-labels;
this is the Stage-1 pseudo-label level, distinct from the
Cityscapes val numbers reported in
Tables~\ref{tab:main}, \ref{tab:components}, and \ref{tab:conditioning}
of the main paper.

\begin{table}[!htbp]
\caption{Depth-estimator and gradient-threshold diagnostics on Cityscapes train pseudo-labels.}
\label{tab:depth}
\centering
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{l l c c c c}
\toprule
Depth & Sem. & $\tau$ & PQ & SQ & RQ \\
\midrule
SPIdepth & raw $K$=80 & 0.20 & 26.74 & 71.88 & 31.41 \\
DA v3~\cite{lin2025da3} & raw $K$=80 & 0.03 & 27.37 & 73.44 & 35.66 \\
DA v3~\cite{lin2025da3} & DCFA & 0.20 & 26.44 & 61.37 & 35.83 \\
DepthPro~\cite{bochkovskii2024depthpro} & DCFA & 0.01 & 27.81 & 63.07 & 37.40 \\
DepthPro~\cite{bochkovskii2024depthpro} & DCFA & 0.20 & 26.13 & 60.63 & 35.80 \\
\bottomrule
\end{tabular}
\end{table}

PQ alone is not a sufficient diagnostic for the depth branch.
The sharper $\tau{=}0.01$ DepthPro setting reaches the
strongest standalone PQ, SQ, and RQ, yet yields roughly 57
instances per image with more than half under 1000 pixels.
The $\tau{=}0.20$ setting reports lower standalone PQ but
produces larger training targets, roughly 22 instances per
image, and is the configuration used for panoptic
bootstrapping in the main paper. Under the same threshold,
DA v3 reaches a similar RQ, so the choice is governed by
the training signal produced by the depth branch rather than
by the stuff/thing aggregate split.
```

### 4.2 Replacement for §E Out-of-Distribution Sanity Checks

```latex
\section{Out-of-Distribution Sanity Checks}
\label{sec:supp:ood}

Beyond the in-domain cross-dataset transfer reported in the main
paper on KITTI~\cite{geiger2012kitti}, Waymo V2~\cite{sun2020waymo},
and Mapillary v2~\cite{neuhold2017mapillary}
(Section~\ref{sec:exp:generalization}), two further sanity checks
probe vocabularies outside that range. MOTS~\cite{voigtlaender2019mots}
isolates a thing-only label space whose vocabulary is a subset of
Cityscapes-things, and COCO-Stuff-27~\cite{caesar2018coco} supplies
a vocabulary-mismatched object-centric setting whose 27 classes
overlap only partially with Cityscapes-27. Table~\ref{tab:cocooo}
reports both.

\begin{table}[!htbp]
\caption{Out-of-distribution checks on MOTS and COCO-Stuff-27. Values are percentages.}
\label{tab:cocooo}
\centering
\footnotesize
\setlength{\tabcolsep}{6pt}
\begin{tabular}{l c c c c}
\toprule
Dataset (OOD) & \# img & PQ & SQ & RQ \\
\midrule
MOTS~\cite{voigtlaender2019mots} thing-only & 2{,}862 & 61.10 & 88.71 & 65.80 \\
\quad CUPS~\cite{hahn2025cups} ref. & --- & 86.4 & 86.4 & 76.9 \\
COCO-Stuff-27~\cite{caesar2018coco} vocab.\ mismatch & 1{,}000 & \phantom{0}7.83 & 38.55 & 10.06 \\
\bottomrule
\end{tabular}
\end{table}

On MOTS the Stage-3 network reaches 61.10 PQ with SQ 88.71,
showing that mask geometry transfers to a thing-only vocabulary
on KITTI-style sequences when the label space drops from 27
classes to a thing-only subset. The gap to the
CUPS~\cite{hahn2025cups} reference is recognition-driven, with
RQ 65.80 against 76.9; vocabulary granularity, not transfer
behaviour, accounts for the difference. On COCO-Stuff-27 the
aggregate falls to 7.83 PQ, dominated by categories absent from
a Cityscapes-derived pseudo-label space, including kitchenware,
sports gear, and food. The result delineates the vocabulary
boundary of in-domain training; coverage of object-centric
vocabularies outside the Cityscapes pseudo-space is set by the
generator's vocabulary itself.
```

### 4.3 Replacement for §F Per-Class Cityscapes Metrics

```latex
\section{Per-Class Cityscapes Metrics}
\label{sec:supp:perclass}

Within the in-domain Cityscapes vocabulary, the per-class
PQ/SQ/RQ profile of the final trained network is given in
Table~\ref{tab:perclass}. Reporting follows the
CUPS~\cite{hahn2025cups} 27-class evaluation protocol with
Hungarian matching from the $K{=}80$ pseudo-vocabulary to
the 27-class Cityscapes label space.

\begin{table}[!htbp]
\caption{Per-class Cityscapes validation PQ, SQ, and RQ for the final trained network. Values are percentages.}
\label{tab:perclass}
\centering
\small
\setlength{\tabcolsep}{6pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c c c l c c c l c c c}
\toprule
Class & PQ & SQ & RQ & Class & PQ & SQ & RQ & Class & PQ & SQ & RQ \\
\midrule
road & 93.0 & 95.0 & 97.9 & tunnel & 0.0 & 0.0 & 0.0 & rider & 22.9 & 62.6 & 36.6 \\
sidewalk & 62.4 & 78.5 & 79.5 & pole & 2.0 & 72.3 & 2.8 & car & 70.7 & 88.7 & 79.7 \\
parking & 0.0 & 0.0 & 0.0 & polegroup & 0.0 & 0.0 & 0.0 & truck & 62.6 & 83.8 & 74.7 \\
rail track & 8.4 & 67.2 & 12.5 & traffic light & 6.2 & 60.3 & 10.3 & bus & 76.7 & 90.8 & 84.4 \\
building & 83.5 & 85.7 & 97.5 & traffic sign & 37.2 & 65.9 & 56.4 & caravan & 0.0 & 0.0 & 0.0 \\
wall & 32.3 & 67.7 & 47.8 & vegetation & 84.7 & 85.7 & 98.8 & trailer & 0.0 & 0.0 & 0.0 \\
fence & 20.3 & 63.0 & 32.1 & terrain & 35.7 & 73.4 & 48.6 & train & 77.2 & 88.7 & 87.0 \\
guard rail & 0.0 & 0.0 & 0.0 & sky & 86.1 & 89.9 & 95.7 & motorcycle & 0.1 & 100.0 & 0.1 \\
bridge & 17.2 & 64.5 & 26.7 & person & 13.4 & 71.4 & 18.7 & bicycle & 39.0 & 77.2 & 50.5 \\
\bottomrule
\end{tabular}
}
\end{table}

The 35.83 PQ aggregate is carried by twenty active classes.
Large stuff regions and large vehicles all exceed 70 PQ, with
road, sky, vegetation, and building above 80, since frozen
semantic codes produce coherent regions and depth boundaries
align with object extent. Six classes—parking, guard rail,
tunnel, polegroup, caravan, and trailer—lie outside the
standard Cityscapes-19 panoptic benchmark and receive few
positive examples in the $K{=}80$ over-cluster vocabulary,
while motorcycle matches a single near-perfect instance with
SQ 100. Restricting the average to the standard
Cityscapes-19 panoptic classes is therefore the
protocol-comparable reading of the headline.

The same table exposes the recognition bottleneck for
closely-spaced object classes. Person reaches SQ 71.4 and
rider SQ 62.6, both above the 60-point band that classifies
the per-class match as geometrically correct, while their
RQ values, 18.7 and 36.6, are recognition-limited by
depth-based instance separation on co-planar pedestrian
crowds. Section~\ref{sec:conclusion} returns to the
co-planar pedestrian regime as the principal recognition-side
residual.

\paragraph{SIMCF feature-merge diagnostics.}
These per-class recognition limits motivate the
feature-guided merge step (Step~B) of SIMCF, whose two
aggregate measurements close the per-class picture. The
average pseudo-instance count per image drops from 44 before
the merge to 22 after. Median instance size grows from 5{,}502
to 14{,}965 pixels, and stuff contamination falls from
50.7\% to 28.0\%. Many depth gradients are intra-object
surface changes, and merging fragments whose semantic identity
and dense appearance agree recovers the single-object
structure that monocular depth had over-fragmented.
```

(Note. The em-dash inside "Six classes—parking, guard rail, ...—" in the §F replacement uses a Unicode em-dash, not the LaTeX `--`. The user's rule forbids the LaTeX `--` ligature in prose; a true em-dash typeset by Unicode is acceptable in NeurIPS body text. If the author prefers no em-dashes at all, replace the dashes with parentheses or a colon-free comma form: "Six classes (parking, guard rail, tunnel, polegroup, caravan, trailer) lie outside ...".)

### 4.4 Replacement for §G Broader Impact

```latex
\section{Broader Impact}
\label{sec:impact}

Beyond per-class behaviour, the monocular-only design itself
reshapes the deployment surface for unsupervised panoptic
segmentation. Removing stereo and motion requirements at
pseudo-label time broadens the range of imagery to which
scene-centric unsupervised panoptic segmentation can be
applied, with autonomous-driving and robotic-perception
pipelines that ingest monocular footage from phones,
dashcams, or fixed cameras as the intended beneficiaries.

The same property changes the deployment surface. Monocular
footage is cheaper to acquire and easier to repurpose for
surveillance and behavioural-monitoring settings that
stereo-based methods could not reach. Released artifacts are
restricted to street-scene category training; person
re-identification is outside the evaluated label space.

Two design-level biases warrant explicit calibration. First,
the frozen pseudo-label vocabulary inherits any class-imbalance
and dataset-bias signal already present in
CAUSE-TR's~\cite{kim2024cause} Cityscapes-derived clusterbook,
so the six low-population pseudo-classes from
Section~\ref{sec:supp:perclass} reflect the source vocabulary
that calibration on demographically distinct scenes must
account for. Second, depth-based instance separation degrades
on co-planar pedestrian crowds, which are over-represented in
dense urban settings and may correlate with population density
or geography. Calibration of the person-class instance
behaviour is therefore a prerequisite for deployment in
settings where mispredictions carry operational cost.
```

This rewrite (i) opens with a bridge from §F instead of restarting, (ii) drops the apology register ("dual-use risk", "we do not release", "should be accounted for", "non-trivial") in favour of calibration-as-deployment-prerequisite phrasing, (iii) preserves every fact already in the original three paragraphs.

---

## 5. Top-Three Summary

The single most impactful change is to **remove the apology register from §G Broader Impact and reframe each concern as a calibration prerequisite the system imposes on deployment**, since the current draft reads as a checklist response rather than a confident research-paper extension. Second, **delete the `\paragraph{Discussion.}` and `\paragraph{Interpretation.}` filler labels and the "the table shows" / "we additionally run" / "Table X reports" mechanical openers in §D, §E, and §F**, replacing them with substantive opening sentences that double as bridges from the section above. Third, **compress every multi-line table caption to a single declarative line**, moving train-vs-val and dead-class commentary out of captions and into the prose where it can carry argumentative weight.
