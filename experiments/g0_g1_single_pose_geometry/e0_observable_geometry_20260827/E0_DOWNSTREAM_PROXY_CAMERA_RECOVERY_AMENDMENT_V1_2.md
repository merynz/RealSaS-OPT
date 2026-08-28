# E0 Downstream Proxy — Camera Recovery Amendment V1.2

Status: **AUTHORIZED APPARATUS CORRECTION BEFORE SCIENTIFIC TRAINING**.

The V1.1 production prep successfully built and published 189/437 assets, then failed on
`asset_de72098ec0bd43ce9fc30c60` before any downstream proxy training. The failure was
`insufficient stable half-extent ratios: 84`. The old estimator divided each camera-frame
coordinate by its grid coordinate after an arbitrary `abs(grid) >= 0.05` cutoff and then
required 128 surviving ratios.

Post-hoc diagnosis showed this is a false apparatus failure, not missing camera information.
V0 has 2393 visible rows but a small screen footprint. A through-origin slope fit over all
finite observable rows recovers `h=0.5400000774096994`; the post-hoc camera authority is
`0.5400000643730164`, absolute error `1.3037e-08`, with native reprojection P95 `0 px`.
`camera.json` remains forbidden in the forward estimator; it was read only for post-hoc diagnosis.

## V1.2 estimator

For each visible row:

```text
q_x = dot(P,right) = h * g_x
q_y = -dot(P,up)   = h * g_y
```

Recover one scalar `h` by robust regression through the origin. Near-center observations
receive naturally tiny leverage through `g^2`; no hard screen-coordinate cutoff exists.
Deterministic Huber IRLS provides outlier resistance. Fail-close authority is native
reprojection P95 plus x/y slope consistency and numerical information energy.

## Firewalls

- frozen 374 train / 59 selection / 4 calibration population unchanged;
- D0/D1/D2 semantics unchanged;
- MUTUAL_P003 unchanged;
- model/optimizer schedules unchanged;
- no downstream scientific training had started;
- no downstream outcomes had been inspected;
- Proxy32 and DEV32 remain closed;
- V1.1 packs are not reusable under V1.2. Resume records are now bound to both the
  downstream builder SHA and `e0_geometry.py` SHA.

Geometry source SHA: `88872577354055e8559e5e343834355068d2cc5bfd2633f7196a6ae45324fa40`  
Prep runner V1.2 SHA: `d5d38fe00537922b5bc87f90a59a897ff46af4411b3ddb34c59369954d13d5da`
