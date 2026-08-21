# RealSaS-OPT

> **ACTIVE RESEARCH WORKSTREAM — 2026-08-21**
>
> Branch: **`g0-g1/single-pose-geometry`**
>
> The active canonical frontier has moved from the older N1D/mechanics continuation to **G0 → G1 single-pose multiview geometry**. For a new ChatGPT session, “GitHub'a bak ve devam et” means: read the active branch's `CURRENT_STATE.md` first, then `canonical/PRODUCT_CONTRACT_V1.md`, then `experiments/g0_g1_single_pose_geometry/`. Do not continue from the stale `main`-branch `CURRENT_STATE.md`.

Private canonical research workspace for RealSaS.

## Storage model

- **GitHub** — canonical code, preregistrations, tests, compact results, reports, CURRENT_STATE and experiment history.
- **Google Drive** — heavy corpora, caches, checkpoints, large proof packs and other data-depot artifacts.
- **ChatGPT Library** — transient / active working artifacts and diagnostics that benefit from fast retrieval.

The three stores may be combined when useful, but authority/provenance must remain explicit. Heavy raw data should not be committed to this repository.

## Continuation discipline

- Active experimental code stays on the named workstream branch until its gate is stabilized.
- The active branch's `CURRENT_STATE.md` is the single handoff authority.
- No prior research component may disappear silently; active/reserve/deprecated status must be recorded with provenance.
- `main` serves as the stable landing page and always points to the current workstream.
