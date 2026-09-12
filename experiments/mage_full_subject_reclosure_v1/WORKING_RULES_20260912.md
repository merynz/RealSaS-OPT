# Working rules for this repair branch

1. Measure before patching.
2. Freeze authority/hash/contract before training.
3. Never let a downstream solver hide an upstream omission.
4. Prefer frozen checkpoint replay to retraining.
5. Preserve historical artifacts; supersede by explicit lineage only.
6. Require component-level accounting whenever aggregate coverage can hide a missing object.
7. No hard-coded Mage names in generic Compiler/product code.
8. No threshold loosening to make a failed run pass.
9. No claim beyond the exact gate just measured.
