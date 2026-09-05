# RealSaS — Mage Geppetto B1s postmortem + optimizer fork V1

## Frozen evidence

B1s structural serialization remained geometry+multplicity only and used direct canonical slot supervision with no Hungarian calls in training. The completed run reached no PASS within 16384. Its best checkpoint was step 10752 with nearest p95 0.0213794168, nearest max 0.0312670656, outside-capture count 3 and occupancy L1 error 3. The historical next check at 10816 catastrophically moved to outside 41/41, occupancy L1 41, nearest p95 0.2717448175 and structural-slot p95 0.7398356199.

B1a remains comparison-only upper reference: best/final nearest p95 0.0039795977, exact occupancy, stable PASS at step 8000. No B1/B1a/B1s result authorizes generalization, count/STOP, mechanical-role or product identity claims.

## Free audit of the three missing controls

The final B1s checkpoint stores the best prediction from step 10752. Auditing that prediction against the frozen structural target gives the exact outside serialized slots:

- slot 34 → original teacher row 25 → locus group 24 (multiplicity 2), slot error 0.029741874; not automorphic;
- slot 35 → original teacher row 26 → locus group 20 (multiplicity 2), slot error 0.031267066; not automorphic;
- slot 36 → original teacher row 27 → locus group 23 (multiplicity 3), slot error 0.021379417; member of one exact size-2 automorphic sibling class.

Thus only one of the three deficits overlaps an unresolved automorphic tie. The other exact size-2 automorphic class is locus group 11 and is fully captured.

The three failures are consecutive slots on the positive-X serialized branch. A structurally mirrored negative-X branch is serialized much earlier and is fully captured, including its automorphic duplicate class. Direct best-checkpoint slot-error pairs are:

- right 33: 0.015574762 vs left 4: 0.000291473;
- right 34: 0.029741874 vs left 5: 0.000668048;
- right 35: 0.031267066 vs left 6: 0.001194247;
- right 36: 0.021379417 vs left 7: 0.002860866;
- right 37: 0.006222545 vs left 8: 0.002003710;
- right 38: 0.001926848 vs left 9: 0.002335113.

This disfavors a pure automorphic-tie explanation and makes late-slot / autoregressive-trajectory difficulty a concrete competing mechanism. It does **not** authorize changing the serializer yet.

## Next diagnostic: B1s optimizer/recurrent-sensitivity fork

Scientific source stays byte-identical to B1s (`8b3eb659abbec3a128a314abc59ad07fdc628329`; runtime SHA `ad6b1b4e32b84ec151fbeaba20ae78c9dfe71fcb1685acf212995a59a4541cc8`; serialized-content SHA `a9915d66adc65dc2c69a97f60b7e29a96a64d82f2a4c10fc78d72148e4768cd3`). No serializer or decoder change is permitted in this fork.

1. Deterministically replay B1s to step 10752. Metric parity and outside slots [34,35,36] are required before forking.
2. Run baseline first. It must reproduce the historical catastrophic state at the same original check step, 10816. If not, abort before intervention branches are run/interpreted.
3. From the same 10752 model+optimizer+RNG state, run four branches through step 11264:
   - baseline: AdamW lr 3e-4, eps 1e-8;
   - global gradient clip: max norm 1.0, otherwise baseline;
   - LR/10: lr 3e-5, eps 1e-8;
   - high-epsilon: lr 3e-4, eps 1e-4.
4. Record dense per-step telemetry: pre/post-clip gradient L2, module gradient L2, sampled Adam `sqrt(vhat)+eps` quantiles, bias-corrected Adam ratio L2, actual parameter-update L2, module update L2, `||Δy_1:41||`, control-state delta, multiplicity/geometry metrics, and selected mirror/deficit slot errors.

Interpretation is conservative. A baseline mismatch invalidates the fork. Clipping failure alone cannot exonerate optimizer dynamics. LR and epsilon branches distinguish global step-size sensitivity from specifically small-denominator amplification. If gradient and parameter-update norms remain ordinary while output-trajectory delta spikes, recurrent sensitivity remains the leading mechanism. A successful diagnostic branch is not a promotion; it must be confirmed in a full preregistered B1s rerun before B2.
