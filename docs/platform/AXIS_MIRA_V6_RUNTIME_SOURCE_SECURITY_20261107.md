# BUSTED — AXIS/MIRA V6 producer runtime source landing (security bounded)

**Audit baseline:** main `173664070e4e2b81d4aa0e79d8c8d4248dbbfe9e`.  
**Upstream exact source reference:** [V6 PR #297](https://github.com/merynz/RealSaS-OPT/pull/297) commit `dd3089957e3126d93db4ad2804e30511cfdca348`.  
**Purpose:** unblock real Rig.v2 / Skin.v2 producer code imports, not claim trained model performance or add an alternative inference owner.

## Runtime source import manifest

Unchanged upstream Git blobs:

| Module | Original Git blob |
| --- | --- |
| `models/axis/rig_program_model_v6.py` | `06e0721d011528cbee909908120f6629e4236072` |
| `models/axis/rig_program_decode_v6.py` | `07e1d10569c542c623675a74a6ef808f44f884c2` |
| `models/axis/scene_conditioning_v6.py` | `6a90ba34255abfeae1785d903c349de79be13e73` |
| `models/mira/skin_fsq_codec_v6.py` | `7e3f59702492a4e84ab19c7208f0d419d20316f0` |
| `models/mira/conditioning_v6.py` | `0af84ececba58d1fd312f5145850170fa8a083a2` |
| `tests/models/test_axis_mira_v6_product_runtime.py` | `0a8bb4c1f44aa102c5ae86abefc7898b9c6f7164` |
| `tests/models/test_mira_v6_skin_fsq_codec.py` | `47a3960aed288a04c91f136cf91f770fddcf8b1d` |

**Deliberate security changes** in two runtime-checkpoint loaders (original pinned blobs `4ae057dcd3d15069ce95b6246000d160dd1522ff` AXIS and `5371984e294463c8b1c96b698eed169d13e245c8` MIRA): change only `torch.load(..., weights_only=False)` to `weights_only=True`. Generated package state remains state_dict tensors + built-in mappings and passes the original save/load parity test. No handwritten pickle allowlist, `weights_only=False` fallback or untrusted import during load is introduced.

Independent security tests use deliberately pickled executable globals and assert loader rejection **without execution**; separate tests reject altered codec binding and compiler-code-repair authority.

## What is actually usable

This is **runtime source code**, not a qualified ModelRelease, production checkpoint, trained prediction, or full worker. It does not import V6 training scripts, change IRIS/TESSA/Corpus370, launch training, alter Graphics or weaken any court.

The original V6 `rig_executor_v2.py` and `skin_executor_v2.py` still need their own reviewed, typed worker integration. The proper contract is:

- AXIS ModelRelease V6 + exact Mesh v1 and RiggingSurface evidence → causal RigProgram + coupled SkinCode proposal → Compiler qualification → CanonicalRig v2 + bound SkinCode.
- MIRA ModelRelease V6 exact codec hash + Mesh v1 + CanonicalRig v2 + SkinCode → decoded carrier skin weights → Compiler simplex/parent qualification → CanonicalSkin v2.
- Mechanics receives exact Mesh v1 / Rig v2 / Skin v2; never remaps v1 to v2.
- Graphics and Motion consume qualified sibling authority later; ProductRevision still requires further proof.

No arbitrary or floating model checkpoint selection; package loader safety does not replace signed/hashed CAS objects, size/resource limits, ModelRelease admission, architecture contracts or controlled execution of model code itself. **ModelRelease, training and product readiness remain blocked until separately verified.**

## Next acceptance gates

1. Dedicated original V6 runtime/FSQ tests and secure loader negative tests must PASS on Python 3.11 + pinned CPU Torch.
2. General `tests/models`, `current-mainline-ci`, model-source ownership and Platform/Go regression gates must remain green.
3. Integrate original Rig/Skin V2 executors as small PRs. Require real artifact-bound CAS inputs and invalid/missing checkpoint fail-closed cases before Go role V2 rewiring. Never register both V1 and V2 under one ambiguous product role or silently change ModelRelease requirements.
4. Only after a qualified V2 output can Mechanics/Motion/deform and Graphics/.rss consumer tests be promoted toward user-facing product.

**Source-code merge is not learning success, and an engineering PASS is not a ProductRevision.**
