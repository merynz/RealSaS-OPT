# Observation / Endpoint Interaction Court V1 — preregistration

**Status:** SEALED BEFORE OUTCOME  
**Authority:** pre-V3.1 mechanism interaction only.

## Question

Do the selected observation/fusion decisions remain jointly usable when presentation evidence is endpoint-bound before compression, or does combining them introduce a hidden incompatibility?

This court intentionally excludes the continuous scene carrier; that is being selected independently by Oracle-Z V5. It isolates the already-selected observation/addressability mechanisms before the final continuous-carrier joint court.

## Synthetic task

- 16 canonical loci;
- 4 orthographic 90-degree-related views;
- source tokens are globally shuffled;
- every canonical locus/view owns independent random local GM bits and independent random alpha/support bits;
- source token carries rotated 2-D position, factorized camera direction, local-detail evidence and the alpha/support observation allowed by the arm;
- canonical query tokens must recover the four per-view GM bits and four per-view alpha bits for every locus;
- a relation query supplies two canonical endpoint indices and must predict XOR of the view-0 alpha/support bits at those endpoints.

No semantic/category shortcut exists.

## Architecture under test

`query-mediated cross-view fusion -> shared canonical locus state -> private GM / GP subspaces -> local heads`

For endpoint relations:

`typed endpoint refs -> gather/query endpoint-local GP state -> bind endpoint A/B payloads -> relation MLP`

The selected observation arm receives patch/locus-aligned raw alpha/support. Camera+UV remains factorized.

## Arms

1. `S2_RAW_ALPHA_BOUND` — selected assembly.
2. `FULL_SHARED_RAW_ALPHA_BOUND` — no private GM/GP adapters; compatibility comparator only (Stage A remains sharing-degree authority).
3. `S2_MEAN_ALPHA_BOUND` — replace local alpha by per-view mean alpha, preserving token/count/camera geometry.
4. `S2_RAW_ALPHA_UNBOUND` — local tasks unchanged; endpoint relation receives only globally compressed GP state + endpoint coordinates.
5. `S2_RAW_ALPHA_BOUND_RANDOM_RELATION` — selected architecture but relation labels are independently randomized; firewall.

Five fixed seeds, matched training schedule. Parameter counts are reported; this court is not a new budget-matched sharing-degree selection.

## Gates

Selected assembly must achieve across five seeds:

- GM local mean >=0.98, min >=0.95;
- alpha local mean >=0.98, min >=0.95;
- endpoint relation mean >=0.98, min >=0.95.

Information-contract ownership:

- `MEAN_ALPHA` must fail local alpha recovery (mean <=0.60) while retaining GM mean >=0.95;
- `UNBOUND` must fail endpoint relation (mean <=0.60) while retaining local GM and local alpha mean >=0.95;
- random-relation firewall must remain in `[0.48,0.52]` mean;
- `FULL_SHARED` is allowed to pass; it must not outperform the selected S2 arm by >=5 points on every required metric. Sharing-degree selection remains the separate Stage A court.

If these gates pass, the selected observation/addressability decisions are interaction-compatible and may enter the continuous-carrier joint court.

## Non-claims

No scene-carrier, real shared-scene, product, Truth Builder, FIT8 or genericity authority is granted here.
