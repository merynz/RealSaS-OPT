# V2 Source Recovery V2 — Behavioral Parity Result

**Verdict:** `SOURCE_RECOVERY_V2_FAIL__DO_NOT_TUNE__DO_NOT_RUN_CORRECTED_0_OF_48`

Recovery source SHA-256: `0747bf3754443053fa5a992f03df4cacddc95f75969572fe4de11253f1e441b4`  
Prereg commit: `3f5260c36f686fddfe024a1aaf976ef2e9f6f5ad`  
Source manifest commit: `a56e53f779ee1284800bc4282112e99c2b0ccc0e`  
Independent GFDR evidence SHA-256: `954302fa7962841179521a49e8c989e44535138f0d733f7cd4d76746062593b0`

## Result

Recovery V2 failed before V2 solver behavior or P1 descendant parity was authorized.

### Gate A — full F16 parity: FAIL

| family | historical expected | Recovery V2 observed |
|---|---:|---:|
| 09908 | 1.0000000000 | 1.0000000000 |
| 11032 | 0.9491525424 | 0.9841269841 |
| 12772 | 1.0000000000 | 1.0000000000 |
| 13203 | 0.9508196721 | 1.0000000000 |
| 14404 | 0.9682539683 | 0.9843750000 |
| 14702 | 1.0000000000 | 1.0000000000 |
| 14758 | 0.9838709677 | 0.9843750000 |
| 15290 | 1.0000000000 | 1.0000000000 |

### Gate B — M256 preflight parity: FAIL

- expected reliable denominator: **492**
- observed reliable denominator: **511**
- expected pooled contain2: **0.975609756097561**
- observed pooled contain2: **0.9941291585127201**
- expected worst-family contain2: **0.9322033898305084**
- observed worst-family contain2: **0.9841269841269841**

Because Gate A failed, the frozen source exited at `SOURCE_PARITY_FAIL`. Gate C and Gate D were **not executed**.

## Infrastructure interruption

The first invocation hit the tool execution limit after producing six deterministic Recovery-V2 family caches. No parity output had been emitted or observed. Execution resumed with the same source SHA and its own V2 caches. No source, helper, threshold, parameter, data rule, or metric changed between start and resume.

## Interpretation

The independently evidenced GFDR `k_neighbors=12` local-scale rule is real pre-existing source evidence, but substituting that rule alone does **not** reconstruct the historical V2 geometry/evaluator population. Recovery V2 is therefore rejected as historical V2 source authority.

The mismatch appears before the relational solver: the reliable population and F16/M256 containment already differ. It is forbidden to tune `k`, reliability thresholds, or other helpers against these observed values. Any Recovery V3 requires new independent source evidence identifying another missing or incorrect historical helper/route.

`sealed21 = CLOSED`  
`external10 = CLOSED`  
Corrected observable-geometry 0/48 = **NOT EXECUTED / NOT AUTHORIZED**.
