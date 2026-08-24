from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from p_formulation_corpus_audit_v1 import LEGAL_GEOMETRY_FIELDS, load_legal_primary_geometry


def main():
    with tempfile.TemporaryDirectory(prefix="pv3_master_geometry_firewall_") as td:
        p = Path(td) / "primary_geometry.npz"
        vertices = np.asarray([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]], np.float32)
        faces = np.asarray([[0, 1, 2]], np.int32)

        # Object-dtype values cannot be loaded under allow_pickle=False. Their presence therefore
        # becomes an executable tripwire proving the auditor never consumes hidden metadata values.
        hidden_object_bomb = np.asarray([{"must_not_be_loaded": True}], dtype=object)
        np.savez(
            p,
            vertices=vertices,
            faces=faces,
            parents=np.asarray([-1, 0], np.int32),
            skin=np.ones((3, 2), np.float32),
            canonical_transform=np.eye(4, dtype=np.float32),
            hidden_object_bomb=hidden_object_bomb,
        )

        vv, ff, meta = load_legal_primary_geometry(p)
        assert np.array_equal(vv, vertices)
        assert np.array_equal(ff, faces.astype(np.int64))
        assert meta["consumed_fields"] == list(LEGAL_GEOMETRY_FIELDS)
        assert meta["hidden_field_values_consumed"] is False
        assert "hidden_object_bomb" in meta["ignored_field_names"]
        assert "parents" in meta["ignored_field_names"]
        assert "skin" in meta["ignored_field_names"]

        # Missing a legal field remains fatal.
        bad = Path(td) / "missing_faces.npz"
        np.savez(bad, vertices=vertices, hidden_object_bomb=hidden_object_bomb)
        failed = False
        try:
            load_legal_primary_geometry(bad)
        except RuntimeError as exc:
            failed = "missing_legal_geometry_fields" in str(exc)
        assert failed

        print(json.dumps({
            "schema": "RealSaS.IRISSinglePoseV2.PFormulationV3MasterGeometryFirewallPreflight.v1",
            "status": "PASS",
            "master_npz_superset_allowed": True,
            "consumed_fields": list(LEGAL_GEOMETRY_FIELDS),
            "hidden_object_dtype_tripwire": True,
            "hidden_field_values_consumed": False,
            "missing_legal_field_rejected": True,
            "optimizer_steps": 0,
            "training_authorized": False,
        }, indent=2))


if __name__ == "__main__":
    main()
