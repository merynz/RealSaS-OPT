# RealSaS — Geppetto Reference-Strength FIT1 Apparatus Freeze V1

Status: `BINDING_COMPANION_TO_FIT1_PREREG__NOT_ACTIVE`

This file freezes execution-only apparatus values that are not semantic product
architecture. It exists specifically so a resource limit cannot be reinterpreted
or tuned after outcomes.

## Frozen execution apparatus

- free-running `resource_step_limit`: **128**
- semantic meaning: **execution budget only**
- product joint/control cap implied: **false**
- native STOP remains the only learned generated-count decision within the
  resource window
- training target decode length: exact current FIT teacher cardinality, derived
  from the generic target builder (currently 22 on Mage)
- diffusion inference robustness seeds: `{11, 23, 47, 89}`
- diffusion default sample steps: model-config frozen value `32`
- GSA target_nodes: `1024`
- GSA normal k: `64`
- GSA visibility_depth_tolerance_norm: `0.02`
- direct tensor support target k: `8`

The value `128` is intentionally much larger than the current 22-control FIT
teacher and is not a product prior. It may only terminate computation if native
STOP never fires inside the frozen window. A future experiment requiring a
larger execution window must preregister that change before outcomes.

## Preflight ordering

Before the first optimizer step, the notebook must successfully execute:

1. exact SHA checks for promoted signed zero-surface, normalized teacher corpus,
   and all eight cameras;
2. real IRIS/GSA reconstruction preflight;
3. exact Mage witness checks: 950 nodes, 2813 GSA relations, 897 observed,
   53 completed, frozen support counts and raster statistics;
4. anonymous mechanical-core target re-derivation in world/source frame;
5. target content SHA check;
6. source compilation and unit tests;
7. no-learned-view-slot assertion.

Any preflight failure makes the run `INVALID_PREFLIGHT`; training must not start.

This apparatus freeze does not activate the experiment, authorize promotion,
or create a generalization claim.
