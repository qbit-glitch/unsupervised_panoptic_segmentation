# MBPS Paper Narration Style Guide
## Derived from CUPS (CVPR 2025), CVPR/ICCV/ECCV Best Papers, and Academic Writing Guides

This guide defines the exact narration style for rewriting the MBPS ICCV 2027 paper. Every instruction here is mandatory.

---

## CRITICAL RULE: NO BOLD TEXT IN BODY

NEVER use `\textbf{}` or `\textbf` in body paragraphs, discussion, or method descriptions. The ONLY places bold may appear are:
- Section/subsection titles (LaTeX handles this automatically)
- Table headers/captions where it is the standard convention
- The paper title

In the CUPS paper, zero sentences in the body text use bold for emphasis. Use `\emph{}` (italics) sparingly for key term definitions on first use. After the first definition, use the term in regular font.

Do NOT use bold for:
- "Key insight:" or "Observation:" labels
- Inline emphasis of phrases
- Names of components or losses
- Contribution items

---

## ABSTRACT (150-200 words, 6-8 sentences)

Structure from CUPS:
1. Define the task in one sentence: "Unsupervised panoptic segmentation aims to partition an image into semantically meaningful regions and distinct object instances without training on manually annotated data."
2. Contrast with prior work (1 sentence): "In contrast to prior work on X, we eliminate the need for Y, enabling Z."
3. State the contribution using italics for the key claim: "To that end, we present the first [method] that [does X]."
4. Describe approach specifics (2-3 sentences): "In particular, we propose..." "Utilizing both X and Y yields a novel approach that..."
5. State result with specific numbers: "Our approach significantly improves panoptic quality, surpassing the state of the art by X.X% points in PQ."

Key rules:
- Use italics for the one-sentence key claim (e.g., "we present the first...")
- End with a concrete, quantitative result statement
- No bold anywhere in the abstract
- No bullet points or enumeration
- Write as continuous flowing prose

---

## INTRODUCTION (1.5 columns)

### Flow pattern (derived from CUPS):

**Paragraph 1: Define the task broadly (4-5 sentences)**
Open with a formal definition of the field. Immediately ground it with practical relevance. Use citations to established work. Example from CUPS: "Panoptic image segmentation [40] is a comprehensive scene understanding task that unifies semantic and instance segmentation."
- Do NOT open with a grand philosophical statement ("Vision is fundamental to...")
- DO open with a concrete, well-defined task statement

**Paragraph 2: Survey progress and identify the gap (5-6 sentences)**
Describe recent advances that make the problem approachable. Then pivot to the limitation. Use the structure: "Recent progress in X has been primarily driven by Y. However, Z remains a fundamental limitation." From CUPS: "Recent progress in panoptic scene understanding has been primarily driven by supervised learning. However, obtaining the required pixel-level annotations for high-resolution imagery is time and resource-intensive."

**Paragraph 3: Frame the opportunity (3-4 sentences)**
State why now is the right time. Describe the promising direction. From CUPS: "A highly promising opportunity lies in approaching panoptic segmentation without any manual supervision."

**Paragraph 4: Critique specific prior work with numbered limitations (6-8 sentences)**
Use explicit enumeration with italics: "Being only the first step, [prior method] has several limitations. *First*, [limitation]. *Second*, [limitation]. *Third*, [limitation]."
This is the CUPS pattern exactly — it uses italicized "First," "Second," "Third," not bold, not numbered lists.

**Paragraph 5: Present your approach (4-5 sentences)**
State the method name and its high-level idea. Connect to a known principle or domain knowledge. From CUPS: "We present CUPS: scene-Centric Unsupervised Panoptic Segmentation. Drawing inspiration from Gestalt principles of perceptual organization—e.g., similarity, invariance, and common fate—we complement visual representations with depth and motion cues to extend unsupervised panoptic segmentation to scene-centric data."
- The approach paragraph should read like a narrative, not a feature list

**Paragraph 6: Contributions as inline numbered list (4-6 sentences)**
Use "Specifically, we make the following contributions:" followed by *(i)*, *(ii)*, *(iii)* inline. From CUPS: "Specifically, we make the following contributions: *(i)* We derive high-quality panoptic pseudo labels... *(ii)* We effectively train a panoptic segmentation network... *(iii)* We demonstrate state-of-the-art..."
- Use italicized (i), (ii), (iii) — NOT \begin{itemize}
- Each contribution should be ONE sentence
- End with a forward-looking statement about additional results

### Introduction anti-patterns to AVOID:
- NO "\noindent\textbf{Key Observations.}" blocks
- NO "(O1)", "(O2)", "(O3)" enumeration
- NO "\noindent\textbf{Our Approach.}" heading
- NO \begin{itemize} bulleted lists
- NO bold labels of any kind
- The three observations should be WOVEN into the critique of prior work and the approach paragraph, not listed as separate labeled blocks

---

## RELATED WORK (0.75-1 column)

Pattern from CUPS and CVPR best papers:

Organize into 4-6 topical paragraphs. Each paragraph:
1. Opens with a topic-defining sentence in the format: "X is concerned with Y" or "X aims to Z" or "X focuses on learning Y."
2. Surveys 4-8 papers in 3-5 sentences, describing what each does concisely
3. Ends by connecting to the present work: "In contrast, we present..." or "Our method differs by..."

From CUPS: "Unsupervised optical flow is concerned with learning optical flow estimation without the need for ground-truth data." Then surveys FlowNet, PWC-Net, SMURF, etc. in 3 sentences. Then: "Current unsupervised optical flow methods (e.g., SMURF [66]) offer accurate flow estimates, fast inference, and generalization to various real-world domains."

Key rules:
- Each paragraph has a topic header in the text (not as a subsection): use a sentence like "Self-supervised representation learning focuses on learning generic feature extractors from unlabeled data, aiming for expressive features that facilitate a broad range of downstream tasks [25]."
- No bold subsection labels within related work — just thematic paragraph breaks
- Actually, CUPS does NOT use bold topic labels in related work. It uses sentence-level topic statements. However, for ICCV format with limited space, using `\paragraph{Topic.}` or `\noindent\textit{Topic.}` is acceptable. BUT the MBPS paper currently uses `\noindent\textbf{Topic.}` which must be changed to `\paragraph{Topic.}` (which the ICCV style renders correctly).
- Actually, looking more carefully at CUPS, the related work does have some formatting but uses running paragraph starts. For the ICCV paper, we can use `\paragraph{}` for sub-topics since ICCV style makes these non-bold or use the pattern of starting each paragraph with an italicized topic phrase.

Recommendation: Use either regular topic-sentence openings (CUPS style) or `\noindent\emph{Topic.}` for sub-topic labels. NEVER use `\textbf{}`.

---

## METHOD SECTION (2-2.5 columns)

### Overall structure (from CUPS):

**Opening paragraph: State the goal and pipeline overview**
"We aim to [do X] without [constraint Y]. To that end, we [approach]. Training and inference is done on [input type] (cf. Fig. 1). Our pipeline comprises [N] stages: *(1)* [stage 1]; *(2)* [stage 2]; and *(3)* [stage 3]."

This is the CUPS pattern exactly. Numbered stages inline, with italicized numbers.

**Subsection titles: Use descriptive step labels**
CUPS uses: "3.1. Stage 1: Pseudo-label generation" and then sub-labels like "1a: Mining scene flow for precise object masks." and "1b: Depth-guided semantic pseudo labeling."

For MBPS, use similar descriptive subsection titles like:
- "3.1. Feature Extraction and Branch Architecture"
- "3.2. Adaptive Projection and Depth Conditioning"
- "3.3. Cross-Modal Fusion via Bidirectional Mamba2"
- "3.4. Stuff-Things Disambiguation from Monocular Depth"
- "3.5. Training Curriculum and Loss Functions"
- "3.6. Panoptic Merging"

### Within each method subsection:

**Sentence 1: State the goal of this component**
"Our first goal is to [what this component achieves]." or "We aim to [objective]." or "Given [input], we seek to [produce output]."

**Sentences 2-3: Explain the approach at a high level**
"Specifically, we [technique] to [achieve goal]." Reference the relevant figure: "(cf. Fig. X)."

**Sentences 4+: Describe technical details naturally**
Weave equations into the text. Don't isolate them as standalone derivation blocks. From CUPS: "We employ SF2SE3 [65], a motion clustering algorithm that uses F and O to fit a variable number of rigid motions, defined in the Lie group SE(3). This results in a set of SE(3)-motions with corresponding masks."

**Key insight sentences: Use natural phrasing**
"Our key insight is that motion and depth provide cues to disambiguate the object instances and semantics in complex scenes." — This appears mid-paragraph, not as a labeled block.

### Method section anti-patterns to AVOID:
- NO "\noindent\textbf{Motivation.}" followed by "\noindent\textbf{Design.}" blocks
- NO "\noindent\textbf{Why X?}" headings
- NO enumerated observations within method subsections
- NO "Evidence:" or "Rationale:" labeled blocks
- NO bold inline labels at all
- Instead, weave motivation naturally: "To address this limitation, we propose..." or "This is particularly relevant because..."

### How to handle equations:
- Introduce equations with flowing prose: "we compute a depth-based weight α at each pixel (h,w) as" followed by the equation
- After the equation, explain what it means: "Note that we add 1 in the denominator for a bounded range, ensuring that α ∈ [0,1]."
- Reference equations naturally: "the loss in Eq. (4)" or "as defined in Eq. (2)"
- Do NOT front-load a section with multiple equations. Interleave explanation and math.
- For already-derived equations, show only the final form in the main paper and cite the supplementary: "(see supp. material for details)"

---

## EXPERIMENTS SECTION (2-2.5 columns)

### Structure (from CUPS):

**4.1. Setup paragraph block** containing:
- "Datasets." paragraph (2-3 sentences per dataset, with specifics: splits, resolution, classes)
- "Evaluation metrics." paragraph (explain PQ, mIoU, AP, and any Hungarian matching)
- "Implementation details." paragraph (pseudo class count, training steps, optimizer, augmentations)
- "Baseline." paragraph (how the baseline was constructed for fair comparison)
- Optionally: "Supervised upper bound." paragraph

**4.2. Comparison to the state of the art**
- Open with scope: "Our experiments assess the unsupervised panoptic segmentation accuracy of [method] within its training domain and its generalization capabilities across diverse datasets."
- Sub-paragraphs for each task/dataset, each opening with the task name followed by a period, like: "Unsupervised panoptic segmentation. Table 1 compares [method] with the state of the art in..."
- Present numbers inline with analysis: "CUPS achieves a PQ of 27.8%, substantially improving over U2Seg (18.4%). This demonstrates how..."
- End the SOTA comparison with a forward reference to ablations

**4.3. Ablation studies / Analyzing [method name]**
- CUPS: "CUPS pseudo-label generation. In Tab. 5, we analyze the contribution of individual pseudo-label generation sub-steps by gradually increasing the complexity."
- Present ablations as incremental: "We start by simply combining the unsupervised semantic predictions and unsupervised instance predictions." Then add components one by one.
- Use sub-paragraphs with descriptive titles for different ablation groups
- End each ablation paragraph with a takeaway: "Every component contributes to the PQ of the labels."

**4.4. Additional analyses** (label-efficient, generalization, per-class, qualitative)

### Experiments anti-patterns to AVOID:
- NO lengthy prose before the first table reference
- NO "Expected Impact" or prediction columns in ablation tables — only measured results (use ___/blanks if not yet evaluated)
- Tables should NOT have bold except in header row
- NO "\noindent\textbf{Ablation Rationale.}" explanatory blocks in the main paper (move these to supplementary)

---

## DISCUSSION (0.5-0.75 columns, or fold into experiments)

Keep discussion focused on 2-3 key points:
1. The central question the paper raises (e.g., "Can monocular depth replace stereo motion?")
2. Limitations (honest, specific)
3. Broader impact or implications

Write as natural flowing paragraphs. No bold sub-headings. Short paragraphs (3-4 sentences each).

---

## CONCLUSION (0.25-0.5 columns)

From CUPS (concise, one paragraph): "We presented CUPS, the first scene-centric unsupervised panoptic segmentation framework that trains directly on rich scene-centric images. Integrating visual, depth, and motion cues, CUPS overcomes the dependence on object-centric training data and achieves significant improvements on challenging scene-centric datasets where prior methods struggle. Our approach brings the quality of unsupervised panoptic, instance, and semantic segmentation to a new level and demonstrates highly promising results in label-efficient panoptic segmentation."

Pattern: 3-4 sentences.
1. "We presented [method], [one-line description]."
2. "By [key mechanism], [method] achieves [main result]."
3. "[Forward-looking statement about implications]."

No bold. No bullet points. No future work section (unless there is space and genuine insight to offer).

---

## SENTENCE-LEVEL WRITING RULES

1. Target 15-20 words per sentence. Vary between 10 and 25.
2. Use active voice predominantly: "We compute..." not "X is computed..."
3. Use passive voice ONLY in methods sections when describing automated processes: "The masks are filtered by..."
4. Place the main idea at the start of the sentence, new information at the end.
5. Never use "clearly," "obviously," "naturally," "trivially," "interestingly," "notably" — these are tell-tale signs of LLM writing.
6. Never use "leverages," "utilizes," "facilitates" — use "uses," "enables," "applies."
7. Use "we" for describing your own contributions. Use passive for others' work.
8. Avoid starting consecutive sentences with the same word.
9. Vary paragraph lengths: some short (2-3 sentences), some medium (4-6).
10. Use em-dashes (---) for parenthetical remarks, not parentheses: "CUPS relies on stereo video---a significant limitation for monocular datasets."
11. Use "e.g.," and "i.e.," with commas after, inside italics for ICCV: `\eg` and `\ie` if the style file provides them.
12. Reference figures naturally: "(cf. Fig. 3)" or "as shown in Fig. 2" — not "See Figure 3."
13. Use consistent tense: present tense for describing your method ("We extract features..."), past tense for experiments ("achieved a PQ of...").

---

## FIGURE AND TABLE RULES

1. Figure captions: Multi-sentence. First sentence describes what is shown. Second sentence explains key details. Third sentence states the conclusion. From CUPS: "Figure 3. Stage 1: CUPS pseudo-label generation. *Instance pseudo labeling* applies ensembling-based SF2SE3 motion segmentation [65] to scene flow extracted from flow and depth estimates. *Semantic pseudo labeling* uses a semantic network, distilling and clustering DINO features [12], combined with a depth-guided inference. *Instance and semantic fusion* aligns the two signals into panoptic pseudo labels."
2. Table captions: One sentence stating what is compared, with any footnotes. "Table 1. Unsupervised panoptic segmentation on Cityscapes val. Comparing CUPS to existing unsupervised panoptic methods, using PQ, SQ, and RQ..."
3. Tables: Use booktabs (\toprule, \midrule, \bottomrule). No vertical lines. Align numbers by decimal point where possible.
4. Place figures and tables at the top of columns (use [t]).
5. First reference to a table/figure should appear BEFORE the float.

---

## WHAT MAKES THIS DIFFERENT FROM LLM-GENERATED TEXT

LLM-generated academic papers have tell-tale patterns. Avoid ALL of these:
1. NO bold emphasis on phrases within paragraphs
2. NO "Key insight:" or "Observation:" labels
3. NO excessive enumeration (O1, O2, O3 or C1, C2, C3)
4. NO grandiose opening sentences ("In the era of..." or "Vision is fundamental to...")
5. NO words like "notably," "crucially," "remarkably," "significantly" used as sentence openers
6. NO repetitive parallel structure across consecutive paragraphs
7. NO over-explanation of obvious connections
8. NO "hedging stacks" like "This potentially may suggest..."
9. Vary sentence structure — mix simple, compound, and complex sentences
10. Use concrete language and specific numbers instead of vague qualifiers
11. The text should feel like it was written by a researcher explaining their work to a peer, not by a system generating a document
12. Read each paragraph aloud — if it sounds mechanical or formulaic, rewrite it
