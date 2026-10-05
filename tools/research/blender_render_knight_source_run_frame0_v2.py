from __future__ import annotations

"""Compatibility entrypoint for the external preset-source RUN frame0 render.

The external FBX may import as one mesh object or many. Object names and object
boundaries are not target equipment authority. Delegate to the v1 renderer,
which now renders all visible source mesh geometry as imported and records the
source only as external motion-preset evidence.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.research import blender_render_knight_source_run_frame0_v1 as base


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return base.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
