# Engine activity process isolation

Temporal owns durable execution and the Go owner records stage results. Each
Python scientific activity now runs in a separate supervised process group.
The Engine parent supplies heartbeats, limits activity concurrency (default one),
and terminates the entire group on cancellation or a failed child. Requests,
results, process logs and supervision receipts remain under the authority root.
The stage result returned to Go is unchanged. Existing version, graph, source,
CAS, input, dependency and court checks still run inside the child.

The previous `asyncio.to_thread` apparatus let different subjects share native
library state. libspatialindex explicitly documents that it is not thread-safe,
including read-only operations:
<https://libspatialindex.org/en/latest/>. Rtree is its ctypes wrapper:
<https://rtree.readthedocs.io/en/latest/>. Process isolation also gives native
crashes and abandoned Blender descendants a concrete lifecycle boundary.

`REALSAS_ENGINE_MAX_CONCURRENT_ACTIVITIES` is an explicit operator capacity
setting, constrained to 1–64. Zero cannot silently enable unlimited concurrency.
The default deployment has one slot and one OpenMP/OpenBLAS thread per process.
Scientific workloads must still have appropriate memory/query bounds; nearest
surface queries now use exact query-independent chunks rather than one unbounded
candidate allocation.

## Recorded cancellation and continuation

An explicit `ResearchCampaignLifecycleRepair.v1` descriptor binds the original
PREPARATION bytes, exact Campaign/release/code and complete Attempt set. Only a
compiled canonical-main Go owner may request cancellation. It first resolves every
member's single delivered command and exact Temporal workflow/run identity,
records the operator reason, requests cancellation, and requires terminal
Temporal witnesses before scoped failure reconciliation. Completed members are
preserved. No artifact, court or previous event is deleted or relabeled.

This lifecycle operation can precede an upgrade while preserving the normal
OPEN-Attempt deployment guard. It does not cancel an unrelated cohort or submit
replacement work. The existing service switch stops the old Engine only after
that guard passes.

A new Campaign may declare `parent_campaign_id` and a separate intervention for
every member. Its exact cohort/roles/order and baseline release must match the
terminal parent. Ordinary ResearchStart then freezes the declared input and
release impact and captures actual baseline artifact IDs. An existing unchanged
artifact must retain its exact identity; an unexecuted parent stage has no artifact
to freeze and may be computed under the unchanged validated contract.

The Corpus370 worker repair is execution engineering, not corpus readiness.
Preserve original Stage47 evidence and all failed/partial measurements. A repaired
transport bundle and explicit child Attempt are required for omitted GLB external
resources. No model training or final truth eligibility follows from deployment.

## Resource admission and external compute

An online runner is not a memory-capacity proof. The Nov2 Corpus370 Engine was
OOM-killed on an8GiB host. Its exact Campaign is terminated through the Go owner,
with completed artifacts and failure history preserved; automatic native
continuation is disabled. A read-only capsule exporter binds the original
294 Attempt/SubjectInput identities and retains76 explicit observation failures.
It copies the pinned evaluator directory and exact input closure, preserving
embedded addresses rather than rewriting scientific JSON. The external worker
uses a distinct COMPUTE namespace and serial, bounded-memory subprocesses.
Each subject's raw evidence is archived to Drive before the next subject.

Notebook computation issues no Registry qualification, final eligibility, split
or training authority. Existing Platform import and independent court validation
remain necessary; the compute exporter is not a stage reuse shortcut. Heavy
work must not be blindly resubmitted on the inadequate native host.
