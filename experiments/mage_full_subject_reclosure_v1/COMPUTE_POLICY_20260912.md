# Reclosure compute escalation policy — 2026-09-12

Use the cheapest execution authority that can faithfully execute the frozen computation.

1. CPU/container/self-hosted: schema, hashes, raster/camera replay, source/component audits, Compiler qualification, GSA, CDT/MWB2, serialization, deterministic geometry tests.
2. Local self-hosted 6 GB GPU: frozen model inference and corrected training only when memory preflight demonstrates no scientific change is required.
3. Colab/A100: only when exact architecture/training cannot execute safely on the local runner without changing scientific batch/sample semantics.

Memory-only query chunking/checkpointing is allowed when it preserves objective, sample schedule, model state semantics and declared numerical tolerance. Scientific batch-size/sample-distribution changes require separate preregistration.
