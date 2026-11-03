"""Read authored glTF bind-rest mesh attachments without Blender-derived truth.

V1 evaluates the default glTF node pose. For an unskinned object parented under
a skin joint, Blender's bind-rest import evaluates that joint in its inverse-bind
rest frame. Reconstruct only that parented object's matrix. Skinned mesh POSITION
remains in its actual raw bind coordinates; do NOT apply skinning a second time.

No raw source bytes or historical V1 reader are modified. Ambiguous joint/skin
bindings fail closed instead of guessing one skeleton.
"""
from __future__ import annotations

import numpy as np

from tools.corpus.gltf_bind_scene_v1 import SourceGLTF, Y_TO_Z

SCHEMA = "RealSaS.SourceGLTFBindRestReader.v2"


def _matrix(value, code):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(code)
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError(code)
    return matrix


def _inverse(value, code):
    matrix = _matrix(value, code)
    if abs(float(np.linalg.det(matrix[:3, :3]))) <= 1e-14:
        raise ValueError(code)
    return np.linalg.inv(matrix)


class SourceGLTFBindRestV2(SourceGLTF):
    """Source reader with uniquely identified source-authored joint bind frames."""

    def _bind_joint_worlds(self):
        doc = self.doc
        # One glTF skin can be instanced by several mesh nodes. A dictionary
        # keyed by skin would silently pick the last and hide frame ambiguity.
        instances = {}
        for node_index, node in enumerate(doc.get("nodes", [])):
            if "mesh" in node and "skin" in node:
                instances.setdefault(int(node["skin"]), []).append(node_index)
        by_joint = {}
        for skin_index, mesh_nodes in sorted(instances.items()):
            if not 0 <= skin_index < len(doc.get("skins", [])):
                raise ValueError("RAW_GLTF_BIND_UNKNOWN_SKIN")
            for mesh_node in mesh_nodes:
                skin = doc["skins"][skin_index]
                joints = list(skin["joints"])
                if len(set(joints)) != len(joints):
                    raise ValueError("RAW_GLTF_BIND_DUPLICATE_JOINT")
                if "inverseBindMatrices" in skin:
                    matrices = self.accessor(skin["inverseBindMatrices"])
                    if matrices.shape != (len(joints), 16):
                        raise ValueError("RAW_GLTF_BIND_INVERSE_MATRIX_COUNT")
                    matrices = matrices.reshape(-1, 4, 4).transpose(0, 2, 1)
                else:
                    # glTF 2.0 defines omitted inverseBindMatrices as identity.
                    matrices = np.repeat(np.eye(4)[None], len(joints), axis=0)
                mesh_world = _matrix(self.world(mesh_node), "RAW_GLTF_BIND_SKIN_FRAME")
                for joint, inv_bind in zip(joints, matrices):
                    if not 0 <= int(joint) < len(doc["nodes"]):
                        raise ValueError("RAW_GLTF_BIND_UNKNOWN_JOINT")
                    authored_bind = mesh_world @ _inverse(
                        inv_bind, "RAW_GLTF_BIND_SINGULAR_INVERSE_BIND")
                    authored_bind = _matrix(authored_bind, "RAW_GLTF_BIND_MATRIX_DOMAIN")
                    by_joint.setdefault(int(joint), []).append(authored_bind)
        for joint, frames in by_joint.items():
            if any(not np.allclose(frames[0], frame, atol=1e-9, rtol=0)
                   for frame in frames[1:]):
                raise ValueError("RAW_GLTF_BIND_AMBIGUOUS_SKIN_JOINT:" + str(joint))
        return {joint: frames[0] for joint, frames in by_joint.items()}

    def geometry(self):
        raw = super().geometry()
        vertices = np.asarray(raw["vertices"], dtype=np.float64)
        bind_worlds = self._bind_joint_worlds()
        offset = 0
        corrected = []
        for node_index, node in enumerate(self.doc.get("nodes", [])):
            if "mesh" not in node:
                continue
            mesh = self.doc["meshes"][node["mesh"]]
            # A skin-owned mesh already stores raw rest-position coordinates in
            # the importer convention. Skinning these again corrupts geometry.
            ancestor = node_index
            while ancestor not in bind_worlds and ancestor in self.parents:
                ancestor = self.parents[ancestor]
            attachment = "skin" not in node and ancestor in bind_worlds
            world = _matrix(self.world(node_index), "RAW_GLTF_BIND_NODE_FRAME")
            if attachment:
                current_joint = _matrix(
                    self.world(ancestor), "RAW_GLTF_BIND_CURRENT_JOINT_FRAME")
                world = bind_worlds[ancestor] @ _inverse(
                    current_joint, "RAW_GLTF_BIND_SINGULAR_JOINT") @ world
            for primitive in mesh["primitives"]:
                count = int(self.doc["accessors"][primitive["attributes"]["POSITION"]]["count"])
                if offset + count > len(vertices):
                    raise ValueError("RAW_GLTF_BIND_VERTEX_LAYOUT")
                if attachment:
                    position = self.accessor(primitive["attributes"]["POSITION"]).astype(np.float64)
                    for weight, target in zip(
                        node.get("weights", mesh.get("weights", [])),
                        primitive.get("targets", []),
                    ):
                        if "POSITION" in target:
                            position += float(weight) * self.accessor(target["POSITION"])
                    updated = (position @ world[:3, :3].T + world[:3, 3]) @ Y_TO_Z.T
                    if not np.isfinite(updated).all():
                        raise ValueError("RAW_GLTF_BIND_NONFINITE_REST")
                    vertices[offset:offset + count] = updated
                    corrected.append({"node_index": node_index,
                                      "parent_joint_index": ancestor,
                                      "vertex_count": count})
                offset += count
        if offset != len(vertices):
            raise ValueError("RAW_GLTF_BIND_VERTEX_COVERAGE")
        raw["vertices"] = vertices.tolist()
        raw["scope"] = "SOURCE_AUTHORED_BIND_REST__NO_SKIN_DOUBLE_APPLICATION"
        raw["reader_schema"] = SCHEMA
        raw["bind_rest_attachments"] = corrected
        return raw
