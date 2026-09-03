# Geppetto

Geppetto is the learned anonymous skeleton/control proposal subsystem.

## Authority boundary

Geppetto consumes admitted `RiggingSurfaceIR`-derived conditioning and proposes anonymous, multimodal control/joint evidence. Its product-facing boundary is `SkeletonProposalIR`.

Geppetto does **not** own canonical joint IDs, the final root/tree/forest, legal graph synthesis, skeleton qualification, product state or proof. Those remain Compiler-owned.

## Current source candidate

Current V2 candidates are mixed into:

`experiments/geppetto_arachne_r6_20260901/`

Known current Geppetto files include the V2 candidate architecture, V2 conditioning, checkpoint, loss, train/eval and capacity apparatus. V1 and legacy files in the same folder are not assumed current; audit must classify them individually.

## Target production layout

```text
models/geppetto/
  README.md
  src/
    model.py
    conditioning.py
    checkpoint.py
  training/
    losses.py
    train.py
  evaluation/
  tests/
```

Teacher projection, synthetic oracle and truth adapters are not automatically model-core code. They enter `training/` only if their information firewall is still valid and they are required by the current training contract; otherwise they remain experimental provenance.

## Compiler firewall

The model may construct `SkeletonProposalIR` objects through the public typed proposal contract. Proposal sequence position, learned root logits, parent logits and support logits are evidence only. Compiler qualification is the only path to `QualifiedSkeletonIR`.