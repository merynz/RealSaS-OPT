# Arachne Mage A0 FIT1 — V7 Replay R1 and persistence incident

Date: 2026-09-09
Scope: V7 continuous forced-field joint-balanced A0 FIT1 only.

## Original V7 scientific result

The original uninterrupted A100 run completed 192 optimizer updates and wrote the sealed JSON result with run fingerprint `18ed583141112c65352d903a97452691cfac5f64383a34ad2a7f0c57ff4abcc6`.

Original final metrics:
- raw row-L1 mean: `1.4641779630781067`
- raw row-L1 p95: `1.8410706227645277`
- deformation-error ratio: `0.8594412207603455`
- continuous pairwise L2 mean: `11.7058` (approx.)
- pre-normalization joint pairwise L1: `0.1039` (approx.)

The uninterrupted run therefore remains a valid scientific trajectory/result, but its final model weights were not durably persisted.

## Persistence incident

The Drive file `ARACHNE_MAGE_A0_SKIN_FIELD_V7_CONTINUOUS_FORCED_FIELD_JOINT_BALANCED_PROGRESS.pt` has the correct V7 fingerprint/config/data binding but contains a step-16 model state. Full re-decode of that state reproduces the sealed step-16 metrics, not the sealed step-192 metrics.

The progress payload is ~3.3 GB because it contains model + AdamW + scheduler + RNG states. The training notebook attempted repeated `tmp -> os.replace()` overwrites on the mounted Drive path at checkpoint intervals. The small final JSON synchronized through step 192, while the large progress artifact remained at step 16. Therefore future large checkpoints must not use repeated direct mounted-Drive overwrite as the sole persistence mechanism.

Policy consequence: the original V7 final model artifact is considered **LOST / NOT RECOVERED**. Do not label any replay state as the original final state unless it reproduces the sealed final metrics within preregistered tolerance.

## Exact-resume replay R1

A recovery notebook loaded the persisted step-16 checkpoint through the original V7 `load_progress` contract, restoring model, AdamW, cosine scheduler, Python RNG, NumPy RNG, Torch RNG and CUDA RNG, then continued steps 17..192.

The replay begins extremely close to the original trajectory:

| step | original mean row-L1 | R1 mean row-L1 | absolute difference |
|---:|---:|---:|---:|
| 32 | 1.828566678 | 1.828447760 | 0.000118919 |
| 48 | 1.827995615 | 1.828063116 | 0.000067500 |
| 80 | 1.688245012 | 1.717543086 | 0.029298074 |
| 112 | 1.547350508 | 1.652147049 | 0.104796542 |
| 144 | 1.474019654 | 1.613238283 | 0.139218629 |
| 192 | 1.464177963 | 1.603641022 | 0.139463059 |

R1 final metrics observed:
- raw row-L1 mean: `1.6036410216113532`
- raw row-L1 p95: `1.8366619539759994`
- deformation-error ratio: `0.9094321131706238`
- continuous pairwise L2 mean: `3.6121952533721924`
- pre-normalization joint pairwise L1: `0.07406528294086456`

The sealed-result recovery gate correctly rejected R1 as a reconstruction of the original final state.

## Causal interpretation

This establishes two separable V7 phenomena.

### A. Branch-sensitive optimization / numerical instability

The restored run starts within ~1e-4 row-L1 of the original at steps 32 and 48, then gradually diverges into a different basin. This is not consistent with a grossly wrong optimizer state, wrong RNG seed, wrong data binding, or wrong resume step. The most plausible current interpretation is amplification of small BF16/CUDA numerical differences by a sensitive optimization landscape.

Therefore `FIT1-inappropriate optimization recipe / numerical basin sensitivity` is now an **observed problem**, not merely a hypothetical risk.

### B. Branch-stable high-p95 wall

Despite large differences in final mean row-L1 (`1.4642` vs `1.6036`) and continuous latent separation (~`11.7` vs `3.61`), both branches finish near the same p95 wall (`1.8411` original vs `1.8367` R1).

This argues that optimization instability alone is unlikely to explain the persistent worst-row failure. A structural/objective/sampling mismatch remains a strong independent candidate.

## Next diagnostic

Treat R1 as an **independent replicate**, never as recovered original state. Run the row/objective autopsy on R1 with explicit label `V7_REPLAY_REPLICATE_R1`:
- raw pre-normalization row mass;
- raw scalar reconstruction vs normalized row-L1;
- pure vs blend rows;
- dominant-joint accuracy;
- active-joint count / entropy;
- expected active-heavy sampler bias;
- logit-space gradient cosine between scalar sampled objective and coupled normalized-row objectives;
- local counterfactual descent probes.

Do not start V8 and do not change FIT1 cardinality before this diagnostic is interpreted.
