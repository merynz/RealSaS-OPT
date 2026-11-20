# RSS.vNext — editor-approved OPEN pin persistence in .rsedit

This change extends the existing editable puppet authoring archive, not the
source .rss or a new cloth solver.

## Two independent identities

- `authoring_revision_id` is unchanged from the immutable Mesh/Rig/Skin
  authoring triplet and original source .rss SHA-256. It is **not** a user
  authentication token.
- `pin_revision_id` seals the user-authored OPEN pin action, the original
  neural REVIEW candidate, selected canonical hem pin vertex IDs, host
  chain and the explicitly prepared (still disabled) XPBD domain hash.
  The pin action does not become native source truth merely because it is
  persisted.

## Authoring flow

1. Create a conventional `.rsedit` Mesh/Rig/Skin revision with
   `compile_rss_vnext_canonical_user_edit_revision_v1`.
   For a pin-only edit with no geometry or skin change, explicitly pass
   `pin_only_authoring_shell=True`. Ordinary no-op calls still fail.
   An unsealed shell has no pin and no XPBD activation authority.
2. Obtain an actual canonical `EditorOpenPinNeuralReviewV1` on the exact
   Mesh/Rig/Skin lineage and a separately user-authored
   `EditorOpenPinAuthoringCommandV1` whose `authoring_revision_id`
   matches the shell. The caller is responsible for a real user action
   and authz. This module never signs or fabricates one.
3. Call `seal_editor_approved_open_pin_rsedit_v1` with the source .rss,
   unsealed .rsedit and two typed records. The Compiler validates the
   complete OPEN connected face component, true boundary pins, distinct
   held/holder faces, host-chain skin consequences and actual canonical
   Mechanics; a repair-needed review ABSTAINS.
4. Store the returned bytes atomically as a new `.rsedit` archive.
   The original source .rss and the old `.rsedit` bytes remain intact.
   The `editor_open_pin.json` entry and its SHA-256 plus the derived
   `pin_revision_id` are sealed by the archive manifest.
5. `reopen_rss_vnext_canonical_user_edit_revision_v1` validates the
   original source archive, original Mesh/Rig/Skin edits, independent
   Mechanics pose replay, all pin identities and policy flags, **re-mints
   canonical Mechanics**, recomputes the XPBD domain and compares the
   domain hash, hem digest and approval receipt with the saved record.
   `editor_open_pin` in its result is either `None` (unsealed) or the
   requalified, **disabled** domain.

This is not source/native pin admission; it cannot set material
truth, source pin qualification, ProductPASS, or automatic XPBD. The
actual caller must provide an explicit `XPBDSurfacePolicyV1` and
activate the requalified domain in `ControllablePuppetSessionV1`.
Post-physics Graphics RGB/alpha authority remains separate.

## Scope boundaries

The web Studio currently persists V4 user-layer edits with its own
authoring-layer schema. The new Python `.rsedit` path is **not yet wired
to a browser approval control or authenticated Studio backend**.
A real production UI must explicitly translate a canonical pin proposal,
require an actual user action and persist this envelope; it cannot accept
the V4 mesh-index-only edits as trusted M* addresses.

The tests use synthetic modifications to a native preview package to
construct an OPEN held surface and a distinct holder. They check byte
identity, save/reopen, independent domain replay, user override,
cross-parent rejection and rehashed false-physics claims. They do **not**
prove that the model has learned cape attachments or that pixels remain
source-faithful after XPBD.
