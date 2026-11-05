# RealSaS Research — Kavya, StdGEN++, Bunraku and generic geometry-only drawable ownership

Checked: 2026-11-05. This is an external competitive-method audit, not proof of competitor internals or a RealSaS model result.

## Kavya (public capability claims; architecture unknown)

Source: https://www.kavya.io/

The product's official page claims single-PNG -> auto cutting/layering, hidden-region inpainting, draw order, deformable meshes, skin weights, bone hierarchy, IK, transform/path and physics constraints, skeletal animation and json/skel/atlas export. It explicitly says its "own AI model" is trained and fine-tuned for 2D animation, cutting, layering, rigging and animating; labels the private beta "launching soon".

The page also mentions "anatomically sensible skeletons" and "motion priors": structural bias is clearly part of the publicly described product. **Do not infer** that Kavya used StdGEN; no public citation, architecture, source code, training roster, reproducible paper or disclosed layer-count prior was verified. In particular, **40–50 layers** is an informal visual impression, not a verified hard cap or internal 40/50-class ontology.

## Real research baseline

- StdGEN (2025 CVPR), https://arxiv.org/abs/2411.05738 : semantic-aware hybrid implicit 3D representation, multi-view RGB and normal reconstruction and semantic component decomposition.
- StdGEN++ (2026 arXiv / IEEE TPAMI), https://arxiv.org/abs/2601.07660 : Dual-Branch S-LRM, semantic surface extraction and coarse-to-fine geometry, layered texture decoding. Its three principal geometric semantic classes are body, clothing, hair, with specialized face texture layers.
- Bunraku (2026), https://arxiv.org/abs/2607.27348 : a structured Live2D character reconstruction system with a **Live2D-aware organ-level taxonomy**, layer diffusion, hidden-region completion, per-layer mesh generation and joint keypose offset prediction; 8,884-model source corpus. It is not ontology-free layer ownership.
- SeparateGen (2026 IEEE TVCG), https://pubmed.ncbi.nlm.nih.gov/41525572/ : generates semantic components body/clothing/hair/shoes.
- LayerAnimate (2025 ICCV), https://openaccess.thecvf.com/content/ICCV2025/html/Yang_LayerAnimate_Layer-level_Control_for_Animation_ICCV_2025_paper.html : automated element segmentation + motion-based hierarchical merging for layer-controlled animation **video**. Its output is not equivalent to a compiler-proven editable puppet.
- Rigby2D (2026), https://rigby2d.com/ : open-source editor AutoRig from **layered PSD** and explicit source layer assets. Valuable editor/export competitor, but that alone does not solve unlayered arbitrary-asset ownership extraction.

## What our V10 truly established

The original source-native 3-character V9 run produced 1104 actual Blender render arms and 36/36 bounded source-vs-soup parity-eligible pose-camera checks. The V10 independent court reused the exact original raw pixel bytes, tested all available nested K=8/16/32/64 candidate group pairs directly, with SHA-verified receipts. CAT 16↔32 was distinguishable in 2 of 4 source-hit tasks; TRICERATOPS 16↔32 was indistinguishable in 2 of 2; COWFISH 16↔32 distinguished in 1 of 3 with very small pixel deltas. Those were previously selected source-hit anchors, **not** a source-wide random, held-out, unseen or author-layer benchmark.

V10 therefore cannot classify *same* or *different* artist drawable, optimize group count, or authorize training.

## V11 new generic heldout experimental design

A source-face group tree is **an experimental geometry partition family**, never a semantic category list or production slot-count prior. Pre-register new source-rest triangle probes from previously untouched fine-group leaves, using only triangle positions/areas, nested group paths and the prior court's *tested-pair unresolved status* for prioritization. Never reuse source mesh/material/bone/action labels. All source addresses remain privileged and the selection algorithm creates no training targets.

For each original source subject, the initial limited budget is up to 12 **previously untested** fine-group leaves. Count actual splittable parent/child branches and how many query probes cover them; report untested branches. In a later real rendering pass, reject a sampled pose/view if source-vs-posed-soup parity, no-op parity, occupancy or native source teacher coordinate identity fails. Do not pre-authorize alpha CAA, mechanical attachment or artist layer identity.

**Genericity success means not just good results on three cartoon animals**. It requires a repeatable subject-independent source pipeline, withheld shape families/animations, per-family provenance and coverage, and a non-anthropomorphic performance audit. Remaining uncertainty can be typed as UNKNOWN/ABSTAIN rather than forced labels.

## Competitive moat phrasing

"Ontology-free geometry and edit-consequence supervision for editable puppet construction" is a plausible differentiator supported by a demanding research agenda. It is **not yet a proved unique-in-market capability or product output**, and is not "prior-free" in an absolute sense: topology, smoothness, nested candidate split/merge operations and renderer controls are inductive biases. The claim becomes defensible once our supervised student succeeds on truly unseen morphology without semantic class/subject-specific author or file-name hints, and authored editability controls remain valid.
