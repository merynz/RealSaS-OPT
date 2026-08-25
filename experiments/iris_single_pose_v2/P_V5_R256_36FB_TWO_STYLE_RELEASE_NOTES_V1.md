# P-V5 R256 36fb × 2 Styles — Release Notes V1

Status: `PREPARED_NOT_RUN__A100_RUN_NEXT`

This release is a hard-asset sufficiency/localization gate following the immutable 8×2 V1 FAIL and immutable +2048 low-LR continuation FAIL.

Key frozen facts:
- fresh R256 model; no checkpoint reuse;
- asset `asset_36fb02305846592b1ecdf3d4` × `cel_clean` + `ink_cel`;
- MAIN `2048 @ 3e-4`;
- fresh-moment TAIL `2048 @ 3e-5`;
- 4096 visible truth loci/view shared exactly across styles;
- no PatchMatch, no camera.json, no new architecture;
- PASS iff both cells satisfy `P_p95 <= 0.005` at one preregistered checkpoint.

Local apparatus validation before release:
- Python AST/compile PASS;
- CPU B=2 full-R P-only forward/backward PASS;
- frozen N/U/Z heads receive no gradient;
- fake 36fb staging PASS;
- fake 4096/view cache PASS with exact shared truth loci;
- parent continuation authority + old two-style PASS verifier PASS;
- scientific optimizer steps during preparation: 0.

User-facing notebook SHA-256:
`eff053232806e762c2a1e88c76109a34e05706a691b8d1e5f70586c4f77939d0`

Exact bundle SHA-256:
`c002cb7510722565fb48d4a311c25616530256e4edd057fe80d4b828beb5ca2a`

The prior identical B=2 R256 GPU preflight passed on a Tesla T4 with `8,115,611,136` peak allocated bytes, so A100 40 GB is comfortably sufficient for this gate.
