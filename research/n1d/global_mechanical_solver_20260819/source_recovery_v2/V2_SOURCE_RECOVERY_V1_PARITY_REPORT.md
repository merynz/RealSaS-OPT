# V2 Source Recovery V1 — Behavioral Parity Result

**Verdict:** `SOURCE_RECOVERY_V1_FAIL__DO_NOT_TUNE__DO_NOT_RUN_CORRECTED_0_OF_48`

Recovery source SHA-256: `578fdddcd17d922603fb463ee1f2cae5b88e0e2c96a850900716abef5e7588be`  
Prereg commit: `0a6b8298f06ad402fcf7f81e76f3858aaa796cdc`  
Parent scientific authority: `08620737ce526b0c0cc6c551f53910e34a1107fe`

## Result

Recovery V1 failed before any V2 solver or P1 descendant behavior was authorized.

### Gate A — full F16 parity: FAIL

| family | expected | observed |
|---|---:|---:|
| 09908 | 1.0000000000 | 1.0000000000 |
| 11032 | 0.9491525424 | 0.9672131148 |
| 12772 | 1.0000000000 | 1.0000000000 |
| 13203 | 0.9508196721 | 0.9841269841 |
| 14404 | 0.9682539683 | 0.9843750000 |
| 14702 | 1.0000000000 | 1.0000000000 |
| 14758 | 0.9838709677 | 0.9841269841 |
| 15290 | 1.0000000000 | 1.0000000000 |

### Gate B — M256 preflight parity: FAIL

- expected reliable denominator: **492**
- observed reliable denominator: **506**
- expected pooled contain2: **0.975609756097561**
- observed pooled contain2: **0.9901185770750988**
- expected worst-family contain2: **0.9322033898305084**
- observed worst-family contain2: **0.9672131147540983**

Because Gate A failed, the fail-closed recovery source exited at `SOURCE_PARITY_FAIL`. Gate C (historical V2 solver behavior) and Gate D (P1 descendant behavior) were **not executed**.

## Infrastructure interruption

The initial invocation was terminated by the tool runtime after seven family caches were deterministically produced. No parity metric had yet been emitted or observed. The invocation was resumed with the exact same source bytes and its own deterministic caches; no source, helper, threshold, parameter, data, or metric rule changed.

## Scientific implication

Recovery V1 is **not** accepted as the historical V2 source authority. The discrepancy appears already in the reliable/F16 geometry population, before relational solver behavior. This result does not authorize choosing another `local_scale`, threshold, or helper by matching the historical numbers. Any new recovery hypothesis requires independent source evidence, a new source authority, three-store persistence, and a new preregistration.

`sealed21 = CLOSED`  
`external10 = CLOSED`  
Corrected observable-geometry 0/48 audit = **NOT EXECUTED / NOT AUTHORIZED**.
