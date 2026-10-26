# V3.1 closure operations — 2026-10-26

## Final closure update

Run 40343827460 completed successfully. Actual scientific reports are CLOSED:
joint V2 12/12 gates, endpoint V2 5/5 gates. Architecture blueprint is frozen;
read IRIS_TESSA_V3_1_ARCHITECTURE_FREEZE.md and ledger V8 for CURRENT state.
The in-flight checkpoint below is retained only as execution history.

The source review also found nine automatic PR research workflows without the
required concurrency cancellation rule. They now cancel superseded PR heads;
scientific push/dispatch executions are not cancelled by that PR-only rule.
This fixes avoidable duplicate stale-check work, not conversation transport.

## Scope and previous in-flight checkpoint

Active research source: research/iris-tessa-v3-apparatus-audit-v1.
Canonical product source remains main at 6eb2af176c06494c9251d9ffe68ba43dfb02219d.
This is the explicitly selected bounded synthetic architecture-search lane.
Real data/checkpoints/training resume through Platform Experiment/Attempt/CAS.

The latest active execution is run 40343827460 on exact source
4740665bf8b07aa06f9c90f416effe9e9997803a.
Workflow: iris-v31-stopgrad-closure-v2.
Do not resubmit while it is running. Inspect its artifacts and scientific
reports; do not infer PASS from a green Actions conclusion.

The run contains:
- full Continuous Joint System V2 with all original six arms and all gates;
- Observation Endpoint Interaction V2 with all original five seeds/arms;
- relation-only gradient-boundary checks before joint optimization;
- complete seed/arm aggregation (partial or duplicate shards fail).

Both V2 treatments change relation-gradient ownership only. V1 apparatus,
seeds, optimizer, data, budgets and thresholds remain sealed. Independent
endpoint seeds run in parallel; no scientific work was removed.
Joint logs report each 1000 steps; endpoint logs report each arm.

## Reconciled previous artifacts

Gradient Ownership V1: CLOSED; STOPGRAD_RELATION selected.
Run 40335120772, source 439ab65bd1f104fa3d615627f1024dbdc426c9c2,
artifact 12656684424; downloaded ZIP digest matches the registry.

Observation Endpoint Interaction V1: NOT CLOSED, despite workflow success.
Run 40320173921, source f18ed5412fd129e51f1a96a05594715925fbfb4f,
artifact 12650701732, ZIP sha256
dd98b35a94a5a08991ead4540d8a628224790357bd3b0416fd70969c05922b1f.
GM/alpha/relation means: 0.8092437744 / 0.8114547729 / 0.9009643555.
Failed selected_assembly and endpoint_binding_information_owner.
This unresolved failure motivated preregistered V2; V1 is not relabeled PASS.

DINOv2 V3.1 Fusion Form V1: CLOSED.
Run 40321235383, source 79cd69d305630ef4ac794b53136dd8bb240da37d,
artifact 12649971130, ZIP sha256
808f60ba5b44cf6031cd897a982d18c1eda3c0382e5fc11c34233a78b9e1209d.
Selected DIRECT_FINAL_384_L2_64_ALPHA64.
Mean local/alpha: 1.0 / 1.0; final cosine mean 0.9974018633,
minimum 0.9960474372; adapter 37248 parameters versus concat 185088.
Frozen foundation source 7764ea0f912e53c92e82eb78a2a1631e92725fc8,
weight sha256 b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9.

## Operational findings

1. Completed architecture courts use GitHub-hosted Ubuntu; there is no
   evidence of an offline self-hosted runner blocking these completed jobs.
2. The 360-minute setting is a runaway guard, not expected runtime or a
   scientific gate. Previous joint/ownership executions took about 12/14 min.
3. Platform-native Oracle-Z PR #218 was explicitly deferred to keep bounded
   synthetic search fast. This is an operations choice, not evidence that Go
   cannot execute the work.
4. platform-cohort-operator runs 40313020039 and 40314459400 are routing/no-op
   observations: real cohort/pretraining jobs were SKIPPED. Their green status
   does not demonstrate that a Platform research Attempt executed.
5. main CURRENT_STATE is product-oriented; active architecture-search memory
   is in this research directory. Freeze/source admission should publish a
   clear entry pointer so reconnecting agents do not repeat branch archaeology.
6. Conversation transport/session failure cannot be diagnosed from these
   workflow logs. The supplied transcript ends with a conversation maximum-
   duration message; that establishes a session limit for that occurrence,
   not a Go/runner failure or a universal explanation for every disconnect.

## Completion rule

Freeze only after full joint V2, observation endpoint compatibility and
DINO fusion prerequisites pass. Freeze means a bounded pre-Truth architecture
baseline and explicit deferred Truth Builder dependencies. It does not mean
trained weights, real-data readiness, FIT8/LOFO/unseen closure or product PASS.
If a V2 court fails, preserve the failure and isolate it in a new preregistered
experiment; never lower its threshold to satisfy the requested milestone.
