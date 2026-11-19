# RSS.vNext post-XPBD Graphics: actual native RGBA parity gate

The product uses a single canonical mesh/rig/skin and graphics UV atlas.
The prior source-to-.rss preview proves independent LBS reopen and raw native
pixel replay. The live final pose bridge additionally writes RSPOSE01/02
from the *real last* Mechanics vertex positions, including optional surface
XPBD.

This gate exercises the **same raw-two-image-derived** canonical
Mesh/Rig/Skin/Graphics/UV preview fixture, creates an explicitly admitted
surface XPBD domain on the canonical mesh, advances actual XPBD timesteps,
and encodes both initial and physically deformed frames via the existing
live pose stream. The unchanged, independently re-opened .rss supplies
graphics.bin + texture.bin while the ephemeral pose.bin supplies the
actual final XPBD positions. The C++ player renders RGBA for both frames.

The gate must prove all of the following:

- Both frames use the identical canonical vertex, face, rig and skin IDs.
- The physically deformed vertex set differs from pure LBS; pinned
  boundaries do not drift.
- The final XPBD state hash appears in the temporary pose receipt.
- The native C++ output contains visible RGBA pixels for both frames and
  the two actual pixel buffers differ when physics deforms the mesh.
- The original .rss archive bytes do **not** change. Saved authoring
  controls remain independently replayable as before.
- Source appearance and hidden surfaces are **not** qualified: the
  virtual preview's pixels are developer candidates, not CAA/ProductProof.

This is a production-runtime parity regression, not source RGB or material
behavioral proof. Source-native hem/pin/M* qualification (#620) and
causal AXIS mechanical-part learning (#613) have separate ownership courts.
Only independently qualified evidence may promote the final product.
