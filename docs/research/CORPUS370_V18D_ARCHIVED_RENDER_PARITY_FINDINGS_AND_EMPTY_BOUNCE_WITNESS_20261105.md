# Corpus370 V18d — exact archived V18b parity diagnosis and targeted empty-soup bounce falsifier

Date: 2026-11-05. **Evidence-only research record; not graphics-truth/model readiness.**  
Parent V18b actual source render run: `42100552844-1` (3 original sources, 2 FAIL in combined edit court).  
Qualified retrospective diagnosis: `42105598958-1` (4/4 jobs PASS, including 3 original-source archived pixel reads).  
Current targeted witness: `42106976442-1` (bounded 2-source 12-frame Blender witness per source). **CAT and TRICERATOPS independent source courts both PASS** (3/3 run jobs completed successfully).  
Branch: `corpus370-v18b-reactive-stain-source-edit-court-v1`, draft PR #343. Do not merge to main or to Corpus370 #327.

## Why we did not rerun 624 edited Blender frames

The original V18b edit court aggregated four logically distinct checks in one failure `V18B_NATIVE_ORIGINAL_RGB_SOURCE_SOUP_NOOP_EMPTY_COURT`:
1. Original source vs pinned V14 original RGBA at heldout pose2.
2. Original source vs source-triangle soup RGBA at same camera/pose.
3. Full soup vs no-op edit.
4. Alpha in all-source-faces-hidden-by-*Transparent BSDF* soup.

The actual original V18b Cat/Triceratops edited images were already sealed privately on Drive even though their composite court failed. `tools/corpus/diagnose_corpus370_v18b_sealed_render_parity_v1.py` independently reopens those **unchanged** exact `NPZ` arrays with the exact V14 manifest and V14 source job hashes. Its criterion remains the original `TOL=0.025`. It measures **raw RGB, premultiplied RGB, alpha, and invisible support** separately, without inventing artist SAME/DIF labels.

## Newly independently measured results: exact V18b original artifacts

| Original source | Source↔V14 (4 views) | Original↔soup (4 views) | Soup↔NOOP (4 views) | Soup-EMPTY alpha <= .025 (4 views) | Observable original single-face edit-proxy cases /24 |
| --- | --- | --- | --- | --- | --- |
| CAT | 4/4 PASS | **4/4 PASS** | 4/4 PASS | **2/4 PASS** | **14/24** (six distinct target faces across views) |
| TRICERATOPS | 4/4 PASS | **4/4 PASS** | 4/4 PASS | **0/4 PASS** | **24/24** (six distinct targets) |
| COWFISH | 4/4 PASS | **4/4 PASS** | 4/4 PASS | **4/4 PASS** | **0/24** (zero qualifying edit reference effects) |

The prior aggregate error was **NOT** source/soup RGB drift or NOOP mismatch. This matters: do not rewrite source appearance, relax tolerance or condemn V18a groupings based on a rendering transport fault.

Specific original EMPTY alpha remnants:
- CAT: view 0=0; view 2=10 pixels, max alpha 0.71875; view 4=0; view 6=10 pixels, max alpha 0.6875. Total 20 unexpected alpha pixels.
- TRICERATOPS: view 0=20 pixels, max alpha 0.3125; view 2=1, max 0.03125; view 4=18, max 0.28125; view 6=1, max 0.03125. Total 40 pixels.
- COWFISH: 0 unexpected EMPTY pixels on all views. 24/24 original single-face edit proxies are truly **unobservable** at the raster/effect tolerance. Their zero edit effect **cannot** serve as negative drawable truth. Coarser group edits are measurable but not ground-truth collateral without an observable reference.

All three original-source diagnostic jobs and independent redteam CI PASS: https://github.com/merynz/RealSaS-OPT/actions/runs/42105598958 . Private exact report SHA readbacks:
- CAT `4180aefd54c1a7bd36e6cd7bd1291c15be36345fa8658c4eb953d5a4968adcd3`
- TRICERATOPS `f1ff96926afe4988f87a0ab737aa91f751bc4161ade03dc17baeac2f7745e116`
- COWFISH `a16f7ed9d5aab0fb84ca025cc8524309eff563670c250bb2fe25907b21f5f311`

## Candidate mechanism — explicit **hypothesis**, not proven

Original V18b never physically removes individual source triangles. It assigns them a pure *Transparent BSDF* material through `_group_mode()`. The original source renderer `probe._configure_measurement(32)` does not raise Cycles's default `transparent_max_bounces`; Blender source lists a default of **8**. Dense overlapping fully transparent source triangle paths can be prematurely terminated by this finite limit. The measured empty-source alpha values occur in 1/32-like increments (32 Cycles samples), consistent with stochastic transparency exhaustion but not independently causal proof.

Sources: https://docs.blender.org/manual/en/4.5/render/cycles/render_settings/light_paths.html ; https://github.com/blender/blender/blob/main/intern/cycles/blender/addon/properties.py .

## Preregistered minimal source-only causal witness

Only two original sources with known EMPTY failures are eligible (CAT and TRICERATOPS), and only previously sealed original pose2 + 4 original cameras, at original 192px and original 32 Cycles samples. No new source geometry or animation is generated.

Three measurements per view, **12 genuine Blender frames per source**:
1. **Original** empty soup: all triangles assigned Transparent BSDF with original `transparent_max_bounces` (expected default 8), to reproduce the bad empty-alpha mask.
2. **High-bounce** empty soup: same original source faces/material policy/camera/sample count and renderer, `transparent_max_bounces=256` (changed explanatory variable).
3. **Physically hidden** empty soup: all soup mesh objects `hide_render=true`, never relying on transparent BSDF.

Independent court reopens exact raw RGB(A) of all 12 frames, verifies parent V14 job and original retrospectively measured SHA, compares originally bad per-view counts against rerun baseline, and recomputes alpha > .025. It reports individually whether increased transparency depth and actual geometry hiding remove alpha remnants.

**If** original reproduces, high-bounces becomes empty, and physical hide becomes empty, the Cycles transparency-path saturation hypothesis is supported. If physical hide becomes empty but high-bounces does not, investigate path / overlapping primitives without lowering the strict tolerance. If original fails to reproduce, no causal closure; inspect render seed and replay state. If physical hide remains nonempty, investigate non-soup scene objects or renderer background. Do not silently turn low-bounce failure into a graphics-family PASS.

Paths:
- `tools/truth_foundry/blender_corpus370_v18b_transparency_bounce_witness_v1.py`
- `tools/corpus/court_corpus370_v18b_transparency_bounce_witness_v1.py`
- `tools/corpus/hosted_corpus370_v18b_transparency_bounce_witness_v1.sh`
- `.github/workflows/corpus370-v18b-transparent-bounce-source-witness.yml`
- `tests/corpus/test_corpus370_v18b_transparency_bounce_witness_v1.py`

Witness run: https://github.com/merynz/RealSaS-OPT/actions/runs/42106976442

## First actual causal outcome: CAT

GitHub source worker in run `42106976442-1` independently finished successfully and remotely SHA-readback verified source court receipt `d31d987a6cf592263503bb96201be95872638822f62b19435d6d42e21125c693`.

The original failed empty-mask reproduced exactly (view 2 = 10 alpha-remnant pixels, view 6 = 10; views 0/4 = 0). With `transparent_max_bounces=256`, **all four views have 0 remnant pixels**. With actual soup `hide_render=true`, likewise **all four views have 0 remnant pixels**. The independent court re-opened 12 raw rendered RGBA frames and reported `transparent_bounce_limit_hypothesis_supported=true`. No artist-owner or model/training authority was minted. This independently supports a **Cycles transparency-path saturation** cause for Cat, rather than bad source/soup correspondence or V18a segmentation.

The second source TRICERATOPS was independently measured and **PASS** on its own 12 actual Blender RGBA frames: view 0 **20→0→0**; view 2 **1→0→0**; view 4 **18→0→0**; view 6 **1→0→0** alpha-remnant pixels (original Transparent BSDF → 256 transparent bounces → geometry physically hidden). The independent court's remotely SHA-readback receipt is `d750452cd3db1f6552909aa7ba893533d266babb3d43d1eb9ea8ba6c2f0e2674`. Thus transparency-path exhaustion is a reproduced causal source of V18b EMPTY alpha failure on **both** affected original subjects, not just an extrapolation from Cat.

The frozen V18b original group-edit images remain **unqualified**: other REMOVE/ISOLATE arms also used the inadequate Transparent BSDF path. A clean physically removed geometry edit canary + independent court are still required; it would be scientifically invalid to promote all old edit rows solely because `SOUP_EMPTY` can be repaired.


## After causal witness — continuation gate

Do not re-run the expensive V18b group edit matrix merely to make CI green.

- If the transparent-material fault is confirmed, replace **only the removal mechanism** with a mechanically equivalent true visibility/mask operation that handles per-face soup groups without losing source UV/material, including single target face and group remove/isolate. Keep original V18b output versioned, retain exact per-view/source parity and independently verify no-op/empty and heldout edits in a *small canary* before full three-source render.
- If layer candidates fail heldout operational edit consequence independently, retain UNKNOWN or negative **operational** constraint with explicit support; do not claim original artist layers.
- Cowfish requires a new *precommitted ray/geometry-only observable target stratification*, not handpicking visible edits after seeing results. Preserve all six originally precommitted unobservable targets in audit denominator.
- Later compare V18a union-find/FH/Mutex Watershed/Lifted Multicut/optional higher-order on identical *valid* edge evidence, pose, budget and heldout editing.
- Do not change IRIS/TESSA head architecture or model-admission flags. This remains teacher-truth **instrument correctness** work.

Final truth status: operational DRAWABLE UNKNOWN; artist SAME/DIF UNKNOWN; full Graphics/CAA, IRIS training, TESSA training, materialization authorization **FALSE**. No new model training and no edit gold minted.
