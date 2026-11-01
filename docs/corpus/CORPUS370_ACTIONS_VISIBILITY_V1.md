# Corpus370 Actions visibility

All active closure work is performed on `alfred/corpus370-training-readiness-closure-v1` and is expected to produce visible GitHub Actions runs through `.github/workflows/corpus370-training-readiness-closure-v1.yml`.

The workflow is intentionally a lightweight source/contract gate. Heavy Drive/Blender materialization remains a later explicit workflow/notebook stage and must publish receipts back into the same authority chain.

Current visible milestones:

- closure contract present and training explicitly closed
- subject × family authority matrix builder present
- fail-closed matrix semantics under test
- next: bind the matrix builder to the recovered 370 source-inspection receipts and emit the first real 370-row working matrix
