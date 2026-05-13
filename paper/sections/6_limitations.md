---
section: "6 Limitations"
status: locked concise v4 (2026-04-29)
paragraphs: 1
words: ~45
notes:
  - Removed single-seed, single-dataset, depth-dependency, and protocol caveats from the main limitations section.
  - Kept only the two substantive limitations: dead classes and object-centric transfer failure.
---

# 6 Limitations

Dead classes remain the main in-domain limitation: six classes in the 27-class pseudo-vocabulary receive too few reliable pseudo-labels, leaving them effectively absent after Hungarian alignment. This vocabulary limitation also explains the main out-of-domain failure. The Cityscapes-derived pseudo-label space transfers within driving scenes, but it fails on object-centric COCO-Stuff-27, where many categories are outside the supervision represented by the learned vocabulary.
