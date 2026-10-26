# Oracle-Z Matched Carrier Result V5

**Run:** `40319721878`  
**Head SHA:** `0a7c3bfa089fcf26a6660ebf5f71a4d8d7ec9ec6`  
**Artifact digest:** `sha256:d97f565b9b98c03221866b005c5ba903b3f9b4bce68a4ae3b0b2aea3151723ea`  
**Status:** `TRIPLANE_RETAINED_PROVISIONAL_TINY_V5`

## Validity

All preregistered apparatus gates passed:

- T/D carrier parameters exactly matched: `1,105,920` total each (`27,648` per scene);
- shared-Q parameters exactly matched: `89,476` each;
- easy target points valid under per-scene Q192/Fourier4 upper bound;
- coordinate-only shared-Q control did not solve eligible scene-specific tasks;
- scene-carrier roll increased error by `~8.25x` triplane and `~8.74x` dense, proving carrier-state dependence.

## Result

Overall mean nRMSE:

- TRIPLANE: `0.13293`;
- DENSE3D: `0.12652`.

The direct upper bound declared six family/scale points target-valid. At the hard support gates:

- triplane supported 1/6 valid points;
- dense supported 4/6 valid points.

However, dense-minus-triplane mean improvements at valid points were modest and inconsistent (`~ -3.7%` to `+12.1%` on surface, mixed on thin), and **no family contained two adjacent target-valid factors where dense both passed, triplane failed, and dense improved mean nRMSE by >=25%**. Therefore the sealed V5 selection rule does not authorize switching the carrier.

## Decision

V5 retains triplane **provisionally**, not finally.

The pattern is threshold-sensitive: dense crosses more absolute support gates despite lacking the preregistered large/repeated advantage. Before V3.1 carrier freeze, run a small optimization-robustness court on the frozen V5 target-valid set with multiple optimizer seeds. The robustness court may confirm triplane continuity or justify dense if a smaller but reproducible advantage survives independent optimization seeds.

No V5 result is interpreted on target-invalid thin points.
