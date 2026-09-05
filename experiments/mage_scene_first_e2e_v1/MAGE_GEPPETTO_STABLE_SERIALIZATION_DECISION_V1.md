# Mage Geppetto Stable Serialization Decision V1

Status: **DIAGNOSTIC / PRE-PROMOTION**

## Causal evidence

- B1 (moving anonymous geometry Hungarian): no multiplicity PASS within 8192; best nearest-locus p95 `0.0374387056`, final occupancy L1 `24`.
- B1a (step-0 Hungarian frozen before any optimizer step): stable multiplicity PASS at step `8000`; final nearest-locus p95 `0.0039795977`, occupancy L1 `0`, outside capture `0`.
- Same decoder, same geometry loss, same seed and same Mage 41-on-31 target. Therefore a stable training correspondence is sufficient to remove the observed B1 obstruction.
- B1a is an oracle upper reference only. Teacher row identity is not product or proposal identity authority.

## Intervention frozen before B1s

Introduce a teacher-only deterministic structural serialization:

1. parent-before-child preorder;
2. roots and siblings ordered by normalized XYZ;
3. exact-XYZ ties use the complete recursive descendant geometry/topology signature;
4. exact automorphic ties remain repeated equivalent content and are never broken using source row index or teacher control ID.

On Mage, the serializer is content-invariant under arbitrary teacher-row permutation. Two residual exact automorphic sibling classes of size 2 remain; because their serialized subtree content is identical, multiplicity is preserved without inventing identity.

## B1s scientific question

Can this generic geometry+topology serialization replace the B1a fixed-row oracle for exact 41-on-31 multiplicity learning under the unchanged Geppetto decoder and geometry loss?

B1s uses direct structural-slot supervision and performs **no Hungarian solve in the training loop**.

Frozen horizon: primary `8192`; if no stable PASS, continue the same optimizer/model state to `16384`.

A finite no-pass result may only mean that this serialization was not sufficient within the frozen horizon. It may **not** be reported as decoder/capacity impossibility.

## Promotion boundary

Even if B1s passes, the result authorizes only promotion of the stable teacher-training serialization mechanism. Count/stop, mechanical-role prediction, parent/root/support losses, generalization and product identity remain untested.
