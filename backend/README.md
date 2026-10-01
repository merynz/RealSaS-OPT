# Python platform package status

The Python `backend/realsas_platform` package was used to make the professional-platform contracts executable while the architecture was being proven.

Per `docs/platform/adr/0003-go-control-plane.md`:

- **Go is the production control-plane language.**
- Python remains authoritative for compiler/ML/research engine work and Python-side Temporal Activities.
- Existing Python product-state code is a reference/parity implementation during migration.
- Do not grow a second production control plane in Python.
- Compiler bridge, model execution, scientific proof and engine-worker code may remain Python when that is the correct boundary.

The migration is intentionally incremental: Go must prove semantic parity before Python reference pieces are removed or reduced.
