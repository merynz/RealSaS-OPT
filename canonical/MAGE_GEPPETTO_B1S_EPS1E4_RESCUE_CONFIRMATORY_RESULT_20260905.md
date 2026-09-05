# Mage Geppetto B1s — AdamW eps=1e-4 Rescue Confirmatory Result

Date: 2026-09-05
Status: DIAGNOSTIC_ONLY / RESCUE PASS / NO PROMOTION AUTHORITY

## Frozen causal question

Starting from the parity-certified B1s step-10752 state, change only AdamW `eps` from `1e-8` to `1e-4`, keep LR=`3e-4`, WD=`1e-4`, model/loss/serialization/fixture unchanged, and run to step 16384 under the original B1s geometry + exact multiplicity gate.

## Authority

- Corrected optimizer fork V1.1 ZIP SHA-256: `a5923f39aecce30e9444a2afb074902779d616f068a18c8fbae678fd8c43819e`
- Rescue confirmatory contract tag: `204a338ab2496ede`
- Rescue result ZIP SHA-256: `7e4d11f3facbd9c765ff97b1a1f76c9f68088679bf42a71717f339d84567c72b`
- Scientific source SHA: `8b3eb659abbec3a128a314abc59ad07fdc628329`
- Runtime SHA-256: `ad6b1b4e32b84ec151fbeaba20ae78c9dfe71fcb1685acf212995a59a4541cc8`
- Serialized content SHA-256: `a9915d66adc65dc2c69a97f60b7e29a96a64d82f2a4c10fc78d72148e4768cd3`

## Result

PASS.

- first stable original-gate PASS: step `12544`
- maximum stable check streak: `39`
- best step: `15790`
- best outside capture count: `0`
- best occupancy L1 error: `0`
- best nearest p95: `0.005286240018904209`
- final step: `16384`
- final outside capture count: `0`
- final occupancy L1 error: `0`
- final nearest p95: `0.007887238636612892`
- final structural-slot p95: `0.007887238636612892`
- final PASS: true

Capture radius remained frozen at `0.016685275360941887`.

## Interpretation

This establishes that `AdamW eps=1e-4` is sufficient to rescue the already-trained B1s step-10752 state and close the original 41-on-31 structural-serialization multiplicity geometry gate with a long stable PASS streak.

This does **not** yet authorize optimizer promotion because the intervention was introduced only after step 10752. It therefore does not establish that training from initialization with `eps=1e-4` follows an acceptable trajectory or preserves the same frozen gate.

The corrected fork plus rescue result jointly support an optimizer effective-step/small-denominator instability as a real blocker in the original B1s trajectory. Gradient clipping alone did not prevent the historical collapse, while increasing Adam epsilon did.

No count/STOP, mechanical-role, generalization, GSA-boundary, or product-promotion claim is made here.

## Next gate

Run **full step-0 B1s structural serialization with AdamW `eps=1e-4`**, with the same frozen architecture, serializer, geometry loss, forced 41 decode steps, capture radius, 8192 primary budget, 16384 maximum budget, and three-consecutive-check PASS rule. Do not make any other architecture or GSA/conditioning change in that run.

Only a full step-0 PASS can promote the optimizer stabilization into the B1s training path. GSA/RiggingSurface boundary ablations remain a separate causal line and must not be mixed into this confirmation.
