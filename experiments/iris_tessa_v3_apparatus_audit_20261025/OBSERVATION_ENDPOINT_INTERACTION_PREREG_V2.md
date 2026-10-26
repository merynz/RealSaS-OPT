# Observation Endpoint Interaction V2 — preregistration

Date: 2026-10-26. Status: SEALED BEFORE V2 OUTCOME.
Authority: bounded observation/endpoint mechanism compatibility only.

V1 run 40320173921 executed successfully but emitted
OBSERVATION_ENDPOINT_INTERACTION_NOT_CLOSED_TINY_V1.
S2 GM/alpha/relation mean accuracy was 0.809244/0.811455/0.900964;
selected_assembly and endpoint_binding_information_owner were false.
The failure is preserved; CI success is not scientific closure.

Hypothesis: the already selected STOPGRAD_RELATION ownership treatment also
removes relation-gradient interference in the observation assembly.
Only scientific change from OBSERVATION_ENDPOINT_INTERACTION_PREREG_V1:
detach the endpoint-local or globally compressed GP payloads before relation
feature construction in ALL arms. Relation MLP stays trainable; local tasks
continue to train shared/private paths. Architecture, initialization seeds,
truth, all five arms, LR, 1200 steps, batch, evaluation and all gates are frozen.
No residual-adapter, capacity, schedule, loss-weight or threshold change.

Independent fixed seeds may execute in five runner jobs. Every original
(seed, arm) pair must appear exactly once before aggregation; partial shards
cannot emit a scientific verdict. Seed/arm order is restored at aggregation.
Progress logging and sharding are operational changes, not treatments.

If V2 fails, keep architecture freeze OPEN and isolate the measured owner in
a separately preregistered court. Do not weaken gates or select full-shared
from this compatibility comparator alone. No real training/product authority.
