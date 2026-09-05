from __future__ import annotations

import base64
from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import math
import sys

import numpy as np

from experiments.mage_scene_first_e2e_v1 import run_mage_geppetto_fit_v1 as fit
from experiments.mage_scene_first_e2e_v1.geppetto_coincident_loss_v3 import GeppettoLossV3CoincidentVectorized
from models.geppetto.v2.geppetto_train_v2 import geppetto_train_step_v2 as _canonical_train_step


PARTS_DIR = Path(__file__).with_name("fixture_parts")
FIXTURE_SHA256 = "60b0b2788b7d37971dcd81886c165cd56eab3da767f9b555b08cc3f267ed1aa8"
EXPECTED_PARTS = tuple(f"v2q_{i:02d}.txt" for i in range(9))
LOSS = GeppettoLossV3CoincidentVectorized()


def _load_fixture_v2q() -> dict[str, np.ndarray]:
    paths = tuple(PARTS_DIR / name for name in EXPECTED_PARTS)
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Mage fixture parts missing:{missing}")
    encoded = "".join("".join(p.read_text().split()) for p in paths)
    if len(encoded) % 4:
        raise RuntimeError(f"fixture base64 length drift:{len(encoded)}")
    raw = base64.b64decode(encoded, validate=True)
    got = sha256(raw).hexdigest()
    if got != FIXTURE_SHA256:
        raise RuntimeError(f"fixture hash drift:{got}")
    with np.load(BytesIO(raw), allow_pickle=False) as z:
        qscale = np.asarray(z["quant_scale"], np.float32)
        if qscale.shape != (2,) or not np.all(qscale > 0):
            raise RuntimeError("fixture quantization contract drift")
        center = np.asarray(z["center"], np.float32)
        half = float(np.asarray(z["half_extent"], np.float32).reshape(-1)[0])
        geom = float(qscale[0])
        normal = float(qscale[1])
        points = center[None, :] + np.asarray(z["points_q"], np.float32) / geom * half
        normals = np.asarray(z["normals_q"], np.float32) / normal
        normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
        bone_heads = center[None, :] + np.asarray(z["bone_heads_q"], np.float32) / geom * half
        bone_tails = center[None, :] + np.asarray(z["bone_tails_q"], np.float32) / geom * half
        out = {
            "points": points.astype(np.float32),
            "normals": normals.astype(np.float32),
            "edges": np.asarray(z["edges"], np.int64),
            "view_mask": np.asarray(z["view_mask"], np.uint8),
            "skin_joint_index": np.asarray(z["skin_joint_index"], np.uint8),
            "skin_weight": np.asarray(z["skin_weight"], np.float32),
            "bone_heads_world": bone_heads.astype(np.float32),
            "bone_tails_world": bone_tails.astype(np.float32),
            "parents": np.asarray(z["parents"], np.int64),
            "center": center.astype(np.float32),
            "half_extent": np.asarray([half], np.float32),
        }
    if out["points"].shape != (950, 3) or out["bone_heads_world"].shape != (41, 3):
        raise RuntimeError("dequantized Mage fixture shape drift")
    groups = Counter(map(tuple, out["bone_heads_world"].tolist()))
    multiplicities = sorted((n for n in groups.values() if n > 1), reverse=True)
    variants = math.prod(math.factorial(n) for n in multiplicities)
    print("MAGE_COINCIDENT_HEAD_MULTIPLICITIES", multiplicities, "TOPOLOGY_VARIANTS", variants, flush=True)
    if variants != 2304:
        raise RuntimeError(f"Mage coincident topology contract drift:{variants}")
    return out


def _train_step_v3(model, optimizer, conditioning, targets, *, loss_fn=None):
    if loss_fn is not None:
        raise ValueError("E2E wrapper owns the preregistered coincident-symmetry loss")
    return _canonical_train_step(model, optimizer, conditioning, targets, loss_fn=LOSS)


# The compact fixture is transport-only. The only scientific challenger here is
# a generic loss-equivalence hardening: exact coincident teacher loci remain
# anonymous, but their finite topology symmetries are evaluated in one vectorized
# tensor instead of V2's <=256 Python-loop enumeration. Model architecture,
# optimizer, targets, shipping inference, compiler qualification and PASS gates
# remain the frozen Mage V1 FIT experiment.
fit._load_fixture = _load_fixture_v2q
fit.FIXTURE_SHA256 = FIXTURE_SHA256
fit.geppetto_train_step_v2 = _train_step_v3


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/MAGE_GEPPETTO_FIT_RESULT_V1.json")
    fit.run(out)
