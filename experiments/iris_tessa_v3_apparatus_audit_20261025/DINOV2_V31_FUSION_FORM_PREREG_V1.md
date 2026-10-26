# DINOv2 V3.1 Fusion-Form Court V1 — preregistration

**Status:** SEALED BEFORE OUTCOME  
**Authority:** observation-token architecture search only.

## Question

Given the already-selected evidence set `DINOv2 L2 + FINAL + patch-aligned RAW_ALPHA64`, should the 512-D V3.1 view token use the learned concat allocation from the alpha court, or preserve the frozen final token directly and spend only the remaining dimensions on early-detail and alpha residuals?

## Frozen source / data

Reuse the exact DINOv2 S/14 authority, canvas/local-detail task and alpha construction from the sealed dual-level and alpha courts.

Token count remains unchanged: one 512-D token per existing DINO patch location.

## Arms

### C — LEARNED_CONCAT_320_128_64

Current selected alpha-court form:

- FINAL `384 -> 320`;
- L2 `384 -> 128`;
- raw patch alpha `196 -> 64`;
- concatenate to 512.

### D — DIRECT_FINAL_384_L2_64_ALPHA64

Final-preserving form:

- exact frozen FINAL 384-D token copied directly into output;
- L2 `384 -> 64`;
- raw patch alpha `196 -> 64`;
- concatenate to 512.

No learned transform may alter the 384-D final slice before the shared downstream court heads.

Both arms therefore emit the same token count and width and use the same downstream local/final-reconstruction/alpha heads. Parameter counts are reported, not artificially padded.

## Gates

Court validity requires exact DINO identity and the concat arm to reproduce prior mechanism quality.

The direct-final arm is selected if across five seeds it achieves:

- local mean >=0.995 and min >=0.98;
- alpha mean >=0.995 and min >=0.98;
- final-feature reconstruction mean cosine >=0.995 and min >=0.99;
- mean final cosine no worse than concat by more than 0.002;
- adapter parameter count strictly lower than concat.

If direct-final misses any gate while concat remains valid, retain learned concat.

## Interpretation

This court selects only the patch-token fusion form. It does not reopen DINOv2 vs DINOv3, token lattice, camera fusion, scene carrier, Truth Builder or product authority.
