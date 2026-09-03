# RealSaS — R6-G Synthetic Oracle-Substrate One-Family Result — 2026-09-03

**Status:** `PASS_U0__PASS_U1__HETEROGENEOUS_U1_AUTHORIZED__COVERAGE_CONTRAST_LOW`

Prereg authority: `canonical/R6_ORACLE_SUBSTRATE_INFORMATION_SUFFICIENCY_PREREG_20260903.md`

Workflow:
- `r6-oracle-substrate-geppetto-one-family`
- run `33777566769`
- job `100722970279`
- runner region `westus3`
- CPU `AMD EPYC 7763 64-Core Processor`

No witness, camera, shell, visibility, optimizer, threshold or stability parameter was changed after preregistration.

## Fixture telemetry

Witness: `branch_blend_4`

- U0 full-surface nodes: `160`
- U1 admitted observation-oracle nodes: `159`
- visible/full fraction: `0.99375`
- U1 support-view min / mean / max: `2 / 3.610062893... / 4`
- U1 DTB-ND1 normal count: `69 / 159`
- U1 DTB-ND1 normal fraction: `0.433962264...`
- U1 observed local relation count: `0`
- hidden-surface completion: `false`
- direct source-mesh consumer input: `false`

The low normal coverage and zero local-relation count are telemetry, not silent substitutions: U1 still went through exact observation evidence, analytic `P=O+dF`, persistence, the current Assembler route and DTB-ND1 where support permitted it.

## U0_REFERENCE_FULL_SURFACE

Shipping Geppetto config hash:
`6506e3764f7219b758d27c0e5d9dc7e7babc16e2616cd568b00da3dd789bbbf4`

Sustained PASS:
- pass step: `96`
- generated controls: `4 / 4`
- final matched MAE: `0.0274217408`
- final matched p95: `0.0376625210`
- unique-radius ceiling: `0.1650422961`
- Compiler-qualified mechanical status: `roots=1, illegal_parents=0, unsupported=0`
- three consecutive product PASS checks: yes (`32,64,96`)

Verdict: `U0 PASS`.

## U1_OBSERVATION_ORACLE_SUBSTRATE

Same shipping Geppetto config and same initialization seed; only substrate information arm changed.

Sustained PASS:
- pass step: `96`
- generated controls: `4 / 4`
- final matched MAE: `0.0098997783`
- final matched p95: `0.0145597877`
- unique-radius ceiling: `0.1719743013`
- Compiler-qualified mechanical status: `roots=1, illegal_parents=0, unsupported=0`
- three consecutive product PASS checks: yes (`32,64,96`)

Verdict: `U1 PASS`.

## Causal interpretation

Under the frozen R6 interpretation:

- `U0 FAIL` is falsified for this witness;
- `U0 PASS + U1 FAIL` is falsified for this witness;
- shipping-observable oracle information was sufficient for the admitted Geppetto task on this synthetic branch witness.

Therefore the preregistered heterogeneous U1 rung is authorized.

## Important limitation — do not overclaim

The eight-view fixture retained `159/160 = 99.375%` of the sampled full surface. Thus this particular U0/U1 comparison has **low coverage contrast**.

It proves that the actual observation/Assembler representation did not break the consumer despite:

- reduced support-view cardinality;
- only ~43.4% DTB-ND1 normal availability;
- zero admitted observed-local relations in this fixture.

It does **not**, by itself, constitute a strong stress proof that arbitrary self-occluded/full-backside surface information is unnecessary.

No prereg parameter will be changed retroactively. If the heterogeneous U1 rung also passes, any additional coverage/self-occlusion stress requirement must be opened as a separately preregistered strengthening gate, not as a rewrite of this result.
