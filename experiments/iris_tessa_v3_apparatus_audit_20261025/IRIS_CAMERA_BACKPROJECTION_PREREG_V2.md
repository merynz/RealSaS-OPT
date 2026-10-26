# IRIS Camera Geometry / Backprojection Court — Preregistration V2

**Date:** 2026-10-26  
**Status:** `SEALED_BEFORE_EXECUTION`  
**Supersedes for this question:** V1 camera court only as apparatus; V1 result remains immutable and inconclusive.

## Question

Before cross-view fusion, should a source token ask a learned additive camera+UV embedding to rediscover orthographic geometry, or should IRIS receive deterministic canonical/world backprojected coordinates?

V1 entangled this geometry question with a learned four-view parity aggregation task. V2 removes aggregation entirely and asks only whether each observation token can recover its canonical locus under held-out camera yaw and orthographic half-extent.

## Data

Canonical domain: 4x4 lattice, 16 locus classes in `[-0.45,0.45]^2`.

For each sample:

- choose a canonical locus `(x,y)`;
- choose yaw and orthographic half-extent;
- derive camera basis `right, up`;
- project to `u = dot(xy,right)/half`, `v = dot(xy,up)/half`.

Train camera domain:

- yaw: `0,30,...,330` degrees;
- half-extent: `0.70,0.90,1.10,1.30`.

Held-out evaluation camera domain:

- yaw: `15,45,...,345` degrees;
- half-extent: `0.80,1.00,1.20`.

Thus evaluation requires interpolation/generalization across unseen camera geometry, not memorizing exact training camera tuples.

## Arms

1. `CURRENT_FACTORIZED_ADD`
   - separate learned camera MLP over `[right,up,half]`;
   - separate learned UV MLP;
   - add in a common latent, then classify canonical locus.

2. `JOINT_RAW_CAMERA_UV`
   - one learned MLP over concatenated `[right,up,half,u,v]`;
   - diagnostic for whether multiplicative geometry is learnable when raw factors meet before compression.

3. `DETERMINISTIC_BACKPROJECTED`
   - zero-parameter geometry: `xy_back = u*half*right + v*half*up`;
   - same learned coordinate encoder/head family as the positive control.

4. `TRUE_CANONICAL_POSITIVE_CONTROL`
   - give the exact authored canonical `(x,y)` to the same learned coordinate encoder/head;
   - validates optimizer/classifier capacity independently of camera inversion.

5. `RANDOM_LABEL_FIREWALL`
   - independent random train/eval class labels using the current-like arm.

## Matched execution

Seeds: `731,947,1217,1597,2017`.

- CPU, deterministic torch seeds;
- AdamW, LR `1e-3`, weight decay `0`;
- 800 steps;
- batch 1024;
- 32768 held-out samples per seed;
- latent width 32; classifier hidden width 64.

Parameter counts are reported exactly. A treatment does not receive extra budget merely to win.

## Gates

Court validity requires:

- true-canonical positive-control mean accuracy `>=0.995` and minimum seed `>=0.99`;
- random-label firewall mean accuracy `<=0.09`;
- all features finite and exact backprojection max absolute reconstruction error `<=1e-5`.

`CURRENT_FACTORIZED_ADD` is sufficient if mean `>=0.98` and minimum `>=0.95`.

`DETERMINISTIC_BACKPROJECTED` is selected over current only if:

- deterministic mean `>=0.995`, minimum `>=0.99`;
- deterministic has no more trainable parameters than current;
- paired mean accuracy gain over current `>=0.05` and positive gain in at least 4/5 seeds.

`JOINT_RAW_CAMERA_UV` is diagnostic. If joint raw passes `>=0.98/0.95` while current fails, the result identifies the additive factorization as the likely information/interaction bottleneck; deterministic backprojection remains preferred if it also passes its stricter gate with lower/equal parameter cost.

If current already satisfies its gate, retain it as the pre-Truth baseline unless deterministic is materially superior under the explicit selection rule. No preference is granted merely for architectural elegance.

## Non-claims

This court does not retest cross-view query-mediated fusion, image foundation choice, real camera calibration noise, perspective cameras, arbitrary input normalization, Truth139, FIT8 or product authority.
