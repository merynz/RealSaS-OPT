# Existing compute package publication

Run41526186943 completed exact294 input export at2026-11-02T21:17:17Z.
The2037365166-byte archive SHA256 is
`4aa855302017162c94d34638d995c03e8019da21fb21bb214d166785d085ace3`.
The original Drive copy was cancelled at22:44:19Z near the90minute job limit;
the folder remained empty. The only preceding rclone message was a shared-client
retirement notice, which does not by itself prove why the transfer made no
visible progress. Do not claim that the notice establishes a credential failure.
Original metadata survived in GitHub artifact13185224022. No new corpus
computation completed during that transfer interval.

`canonical/CORPUS370_COMPUTE_EXPORT_V1.json` now binds the existing local package,
its exact code/archive/seal/size and original PREPARATION. The publisher validates
these bytes and stages hard links in a new publication directory. It never calls
the scientific exporter or submits a Campaign. The cancelled export directory and
all existing Attempts remain unchanged.

The workflow preserves the complete package as an authenticated GitHub artifact
before Drive publication, allowing a service-specific transport fallback if the
native rclone path fails again. The Drive command reports progress every15seconds,
limits connection/idle waits, bounds retries, and checks the remote archive before
publishing COMPUTE_CAPSULE_RECEIPT.json last. Metadata and the current notebook can
be inspected sooner, but their presence alone is not readiness. Every failure
records PUBLICATION_FAILED__PACKAGE_PRESERVED in PUBLISH_PROGRESS.json.

The regenerated notebook uses corrected self-contained RAM/venv/retry control
cells while scientific code remains pinned to7a61c575b780cf99844fad3f5edc596f1ed8b754.
The capsule is294 measurement inputs plus76 preserved preparation failures;
these are not final IRIS/TESSA eligible counts. Import, independent qualification,
global sign/graphics, head coverage, lineage split and final materialization
remain separate. No optimizer is authorized.

Source impact: the new publisher is called by the publication workflow and its
transport tests only. It does not participate in an Engine stage's implementation
closure or alter the released scientific plan. Notebook generation consumes the
existing sealed receipt; it does not rewrite capsule inputs.
