# Timeout-Era Forensic Index — corrected 2026-10-01

> NON-AUTHORITATIVE agent reconstruction. Repository history, CI evidence, and explicit user decisions outrank this note.

## Executive finding

The normalized 46-stage lineage **was promoted to `main`**.

Exact normalized candidate:
`92de05292b05e4ae8aaab3eed53c515b5eef5929`

Exact merge commit pushed to `main`:
`f459fe5491e367e9e5da07cfc5b1886f1b138ae4`

The merge tree is byte-equivalent to candidate `92de052...` except for ancestry.

GitHub Actions then executed `f459fe54...` as `head_branch=main`. Mainline, runtime, VF23, model-source, proof-service, completion-audit, subject-free orchestration, VF10 and VF13 workflows were green; native source seal was red.

## Why main was later rolled back

The normalized lineage later reached `3529344ef3045e0f2d634b175aae2aed6ec05536`.

It was intentionally preserved as:

`backup/main-normalized-20260930-3529344`

Then `main` was force-updated back to `e6c91184...`.

This was **not an engineering rejection** of normalization. It was a user-directed **safety rollback** because ChatGPT recovery-polling timeouts/context loss were making continued agent-driven main mutations unsafe. The explicit purpose of the backup was to keep the normalized state available for re-evaluation once the platform issue stabilized.

Therefore:

- rollback != falsification;
- rollback != architectural rejection;
- rollback != evidence that old main was technically superior;
- normalized lineage remains a strong candidate/evidence source;
- re-promotion still requires exact current validation because independent blockers were already known.

## Exact rollback evidence

Live Authority job `36766305379`, job `110061221489`, observed:

```text
[new branch] backup/main-normalized-20260930-3529344 -> origin/backup/main-normalized-20260930-3529344
+ 3529344ef...e6c91184d main -> origin/main  (forced update)
```

The same job then generated `a8002b20...` on top of `e6c91184...`.

On 2026-10-01, creating `alfred/repo-memory-map-20261001` triggered the repository's branch-create `Live Authority Context` workflow, which added another generated-only main refresh `c3b1b09c...`. No compiler/runtime production source changed in that side effect.

## Normalized candidate quality boundary

The normalized candidate is substantial and well-tested, but not fully closed:

- Stage18 establishes mechanical/visual mesh ownership split.
- Stage37/38 carry and verify visual-mesh binding identity.
- Stage42 does not execute typed source-owned visual presentation geometry; it fails closed.
- Dedicated source-owned visual wiring gate was red.
- Full candidate gate later passed because that repository wiring test was outside its suite and the runtime seam was explicitly fail-closed.
- Self-hosted forensic run `36844455739` confirms exact candidate `92de...` still fails the repository wiring contract as currently written.
- Actual merged main `f459...` had native source seal drift: `runtime_v2_caa_reference.cpp` 40003 bytes vs sealed 39694.
- Quaternius external motion qualification separately fails `MOTION_MATERIALIZER_DISTINCT_TAKES_COLLAPSED`.

So the correct task is **reconstruct and revalidate normalization**, not blindly force `main` back to the backup.

## Post-merge Knight branch

`ops/current-main-knight-render-20260930` descends from normalized main.

Most of its commits are witness/run-root/rebind/render operations. Two source commits deserve separate treatment:

- `25e765e8...` changes G3B skin-topology stress from a 120-degree metadata/default fallback to the maximum rotation admitted by the sealed deformation envelope.
- `16b8cb72...` adds a unit proof that the sealed envelope wins over metadata fallback.

Workflow `36765075591` proves:
- relevant unit set: **7 PASS**;
- Stage35 diagnostic after the patch: `g3_failures=[]`;
- the run still fails later at `G5_MULTIVIEW_COMPONENT_COVERAGE_FAIL` with 223 failed cells.

This means the Stage35 change is **not a product-pass proof**, but it is a plausible generic correctness fix whose immediate G3 failure mode disappears and whose remaining blocker moves to G5. It should be revalidated subject-free before any promotion.

## Reconstruction rule

1. Treat current rolled-back `main` and normalized backup as preserved evidence states.
2. Treat the rollback as a safety action, not a technical verdict.
3. Classify normalized promotion slices by exact source/test/gate evidence.
4. Separate witness-specific Knight ops from generic fixes.
5. Revalidate generic candidates against current subject-free contracts.
6. Repair/resolve Stage42 visual carrier, wiring-test contract drift, native source seal, and external motion-take qualification.
7. Only then propose a new canonical recovery head.
8. Do not force-move `main` during reconstruction.
