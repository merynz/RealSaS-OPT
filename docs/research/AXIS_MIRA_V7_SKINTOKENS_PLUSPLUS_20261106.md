# AXIS–MIRA V7 / SkinTokens++ — Mechanics-first research contract

**Date:** 2026-11-06  
**Status:** `RESEARCH_CODE_ONLY__NO_TRAINING_AUTHORITY__NO_PRODUCT_MODELRELEASE`  
**Parent:** V6 product branch `axis-v6-rig-program-product`; isolated PR stack.  
**Goal:** compiled puppet is natively controllable ("You can play with your puppet").

## Ownership is non-negotiable

| Domain | Authority | What it owns |
|---|---|---|
| Shape / geometry | IRIS + TESSA / qualified Mesh | shared 3D evidence and mechanical carrier geometry |
| **Rig/Skin learning** | **AXIS–MIRA V7** | skeleton and rest geometry; frame/DOF; controls and relations; compact skin influence fields |
| **Mechanical execution** | **Mechanics Compiler/runtime** | constraint solving, IK/FK, skin deformation, contact/limits, attachment transforms, typed residuals |
| **Graphics** | **IRIS-linked Graphics domain + Graphics Compiler** | CAA/appearance, layer decomposition, occlusion, draw order, visibility, 2D presentation |
| Motion | Control-space motion | editable control trajectories executed by Mechanics |

The Mechanics result may be consumed by Graphics, but neither AXIS–MIRA nor Mechanics owns graphics decisions. No AXIS model target for alpha/layer/RGB/draw order is permitted.

## SkinTokens++ — upstream baseline and what *plus* must mean

Researched the **official source**, not just the paper. Reference repository: https://github.com/VAST-AI-Research/SkinTokens.
Source files reviewed on 2026-11-06:

- `src/model/tokenrig.py` — GitHub blob `ab35e3fb8dd850737cba7d1782f03c75846971be`: Qwen3 0.6B-style autoregressive model; learned Michelangelo mesh conditioning; vocabulary switch grammar; frozen skin VAE in TokenRig.
- `src/model/skin_vae/autoencoders/skin_fsq_cvae_model.py` — blob `6b66d9e9486d261f0a5d67f795bbe60499fb26e3`: separate skin and conditional attention encoders, query sampling, latent quantization, learned decoder with sampled conditioning.
- `src/model/skin_vae/autoencoders/FSQ.py` — blob `9f7bcf0e670071a04a19495ea8efa7d3ead0a1fe`: differentiable round-STE finite scalar quantization, integer-code conversions.
- `src/model/skin_vae_model.py` — blob `a2b4e88b6997883ce5861333bb498dc9593ff80d`: token compression / decompression, per-bone examples, sampling and training wrapper.
- `src/tokenizer/spec.py` — blob `d4cc653519937bb98d6059d765686e2ea53b89dc`: structural skeleton sequence and parent-link tokenization.

Their published checkpoint naming includes `skin_vae_2_10_32768`. **Our 64,000 vocabulary is a new local configuration, not evidence of checkpoint or tokenizer compatibility**. Never suggest that official weights load directly. Respect repository/model licenses and provenance independently.

### Mechanistic comparison

| Capability | Upstream TokenRig / SkinTokens | RealSaS V7 initial implementation | Next step |
|---|---|---|---|
| Unified skeleton + skin sequence | Yes | Yes, via typed V6 wire reused by V7 | new event grammar for interaction-affordance targets only after teacher seal |
| FSQ + straight-through reconstruction | Yes | V7 MIRA uses true gradient-carrying code path | verify with reconstruction-only backward court |
| Spatial attention-based VAE | Yes | Two V7 options: baseline uses weighted pooling; new **learned-query local attention candidate** encodes active sampled support. Not exact upstream CVAE parity | compare baseline and new attention candidate under equal train/validation budgets; add geometry-conditioned decoder attention only if measured improvements justify it |
| Rig conditioning | Learned mesh encoder | Current IRIS/TESSA-adaptable 47-D surface evidence seam | first-class versioned shared-scene adapter, no hidden oracle truth |
| Skeleton generation | Yes | V6 RigProgram language extended by large V7 decoder | require true free-running unseen tests |
| Control rig / interaction affordances | Not the core TokenRig target | V6 control/constraint program + runtime implicit pick from declared controls | learned/evidence-grounded affordances; local DOF/limit solver |
| Real-time deformation | Downstream use after rigging | V6 Mechanics solver; new pick selection bridge | batched/C++ runtime; contact and joint-limit courts |
| 2D drawing/occlusion | Not a TokenRig output | **Not owned here** | downstream Graphics consumers only |

## Implementation in this V7 research branch

- `models/mira/skin_fsq_codec_v7.py`: 64k code vocabulary from (8,8,8,5,5,5), 4 tokens/joint, large carrier ceiling, sparse-aware reconstruction, teacher-relative edge gradient loss, reconstruction-only STE gradient. **Initial codec baseline, not full official architecture.**
- `models/mira/skin_attention_codec_v7.py`: source-inspired but independently implemented learned-query per-joint attention with bounded active-support top-k sampling, inherited straight-through FSQ decode. Not weight compatible with upstream.\n- `models/axis/rig_program_model_v7.py`: 16 x 768, 12-head conditional causal model; typed RigProgram + SkinCodes, tied embedding/output projection, sparse skin logits at SKIN_CODE positions, V6 wire contract and variable morpologies.
- `compiler/realsas_compiler_core/mechanics_interaction_v1.py`: generic barycentric hit-to-declared-control scoring over qualified skin/rig. This is runtime interpretation of already generated structure; it **does not** synthesize control rig, choose new rig parents, or make Graphics decisions.
- `tests/models/test_axis_mira_v7_research.py`: reconstruction-only gradient, hard-code parity, sparse CE/backprop, read-only pick, fail-closed lineage.

## Corpus370 policy

**370 is a candidate acquisition universe, NOT 370 qualified training examples.** Existing PR #327 retains 352 observation PASS / 18 original mask FAIL while further family courts, source-locus ownership evidence, mechanical target qualification and lineage-safe splits remain open. Do not train V7 simply because source observations pass.

V7 requires separated versioned receipts:

1. (Mesh, scene, Rig, Skin) source-to-canonical mapping with evidence and no privileged source-name leakage.
2. Mechanically intervention-tested joint/frame/DOF labels; absence/ambiguous labels typed as masked/abstain, not guessed.
3. Exact MIRA V7 codec TRAIN split targets; no cross-family leakage or unseen access.
4. Frozen MIRA codec **before** AXIS SkinCode teacher targets.
5. AXIS V7 frozen TRAIN/internal validation then sealed one-time UNSEEN scope.
6. Free-running **behavioral** tests: reachable handle IK, hand/weapon attachment, contact consistency, joint-limit compliance, deformation distortion, sparse influence, presentation-agnostic under motion.
7. Knight M+C AA regression independent of unseen selection. No product certification based on a single sample.

No V7 training or ProductRevision promotion is authorized by this research branch.

## Milestones (large increments, no model-size theater)

**A — engineering research candidate (this PR):** higher-capacity coupled Rig/Skin model, improved V7 baseline codec, typed Mechanics hit interaction, CI, exact upstream code references, versioned corpus gate.

**B — architecture science:** benchmark new local-attention MIRA vs initial pooling baseline, then evaluate deeper conditional decoder/encoder parity against upstream; quantify active-support sampling on nonhumanoid corpus, gradient preservation, smoothness and reconstruction at 32k vs 64k, fixed budget.

**C — mechanics semantics:** local DOF/limits, targets/attachments orientation, multi-constraint solves, influence-to-control ambiguity and abstention, spatial interaction tracking; no graphics authority creep.

**D — data/model training:** qualified Corpus370 train/unseen split, frozen codec, coupled AXIS, checkpoint selection only on TRAIN-internal validation, independent unseen behavior court.

**E — real-time product:** C++ cached solver/deform path, mouse interaction latency/p95 metrics, Graphics consuming Mechanics state, independent source fidelity and dynamic visual proofs.

## Research falsification rules

- Strong SkinCode MAE but bad contact, fold or vertex distortion is **FAIL** for playable puppet.
- A legal skeleton with no usable declared handle is **FAIL** for interactive suitability.
- One Knight oracle control-space parity is regression only, never proof of learned rigging.
- A Graphics fidelity problem belongs to Graphics and its evidence binding, not to an AXIS graphics head.
- Any missing ModelRelease, absent exact codec hash or unsealed subject family is fail-closed.
- Bigger architecture is not itself a gain; model capacity must be backed by training data, sampling, pretraining or demonstrable learning curves.

