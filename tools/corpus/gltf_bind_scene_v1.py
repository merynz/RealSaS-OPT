"""Independent raw glTF bind geometry reader; never uses the Blender importer."""
from __future__ import annotations
import base64
from pathlib import Path
from urllib.parse import unquote
import numpy as np
from scipy.spatial.transform import Rotation
from tools.observation.source_scene_contract_v2 import _json_and_chunks

DTYPES = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
Y_TO_Z = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


class SourceGLTF:
    def __init__(self, path):
        self.path = Path(path)
        self.doc, chunks = _json_and_chunks(self.path)
        binary = [data for kind, data in chunks if kind == 0x004E4942]
        self.buffers = []
        for row in self.doc.get("buffers", []):
            uri = row.get("uri")
            if uri is None:
                data = binary.pop(0)
            elif uri.startswith("data:"):
                data = base64.b64decode(uri.split(",", 1)[1])
            else:
                p = (self.path.parent / unquote(uri)).resolve()
                if not p.is_relative_to(self.path.parent.resolve()):
                    raise ValueError("RAW_GLTF_RESOURCE_ESCAPE")
                data = p.read_bytes()
            if len(data) < row["byteLength"]:
                raise ValueError("RAW_GLTF_TRUNCATED_BUFFER")
            self.buffers.append(data)
        self.parents = {}
        for i, row in enumerate(self.doc.get("nodes", [])):
            for child in row.get("children", []):
                if child in self.parents:
                    raise ValueError("RAW_GLTF_MULTIPLE_PARENTS")
                self.parents[child] = i
        self.worlds = {}

    def view_array(self, view_index, count, width, dtype, offset=0):
        row = self.doc["bufferViews"][view_index]
        dtype = np.dtype(dtype)
        stride = row.get("byteStride", width * dtype.itemsize)
        return np.ndarray((count, width), dtype=dtype, buffer=self.buffers[row["buffer"]],
                          offset=row.get("byteOffset", 0) + offset,
                          strides=(stride, dtype.itemsize)).copy()

    def accessor(self, index):
        row = self.doc["accessors"][index]
        width, dtype = WIDTHS[row["type"]], DTYPES[row["componentType"]]
        if "bufferView" in row:
            result = self.view_array(row["bufferView"], row["count"], width, dtype,
                                     row.get("byteOffset", 0))
        else:
            result = np.zeros((row["count"], width), dtype=dtype)
        sparse = row.get("sparse")
        if sparse:
            ix = sparse["indices"]
            indices = self.view_array(ix["bufferView"], sparse["count"], 1,
                                      DTYPES[ix["componentType"]], ix.get("byteOffset", 0)).ravel()
            val = sparse["values"]
            result[indices] = self.view_array(val["bufferView"], sparse["count"], width,
                                               dtype, val.get("byteOffset", 0))
        if row.get("normalized") and result.dtype.kind in "iu":
            limit = np.iinfo(result.dtype).max
            result = np.maximum(result.astype(np.float64) / limit, -1.)
        return result

    def world(self, index, stack=()):
        if index in self.worlds:
            return self.worlds[index]
        if index in stack:
            raise ValueError("RAW_GLTF_PARENT_CYCLE")
        row = self.doc["nodes"][index]
        if "matrix" in row:
            matrix = np.asarray(row["matrix"], dtype=np.float64).reshape(4, 4).T
        else:
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(row.get("rotation", [0, 0, 0, 1])).as_matrix()
            matrix[:3, :3] *= np.asarray(row.get("scale", [1, 1, 1]))[None]
            matrix[:3, 3] = row.get("translation", [0, 0, 0])
        if index in self.parents:
            matrix = self.world(self.parents[index], (*stack, index)) @ matrix
        self.worlds[index] = matrix
        return matrix

    def geometry(self):
        vertices, faces, meshes = [], [], []
        offset = 0
        for node_index, node in enumerate(self.doc["nodes"]):
            if "mesh" not in node:
                continue
            mesh = self.doc["meshes"][node["mesh"]]
            for primitive in mesh["primitives"]:
                if primitive.get("mode", 4) != 4:
                    raise ValueError("RAW_GLTF_NON_TRIANGLE_PRIMITIVE")
                position = self.accessor(primitive["attributes"]["POSITION"]).astype(np.float64)
                weights = node.get("weights", mesh.get("weights", []))
                for weight, target in zip(weights, primitive.get("targets", [])):
                    if "POSITION" in target:
                        position += weight * self.accessor(target["POSITION"])
                world = self.world(node_index)
                transformed = position @ world[:3, :3].T + world[:3, 3]
                transformed = transformed @ Y_TO_Z.T
                indices = (self.accessor(primitive["indices"]).ravel() if "indices" in primitive
                           else np.arange(len(position))).reshape(-1, 3)
                vertices.extend(transformed.tolist())
                faces.extend((indices + offset).tolist())
                meshes.append({"node_index": node_index, "vertex_count": len(position),
                               "skin_index": node.get("skin"), "material_index": primitive.get("material")})
                offset += len(position)
        joints = {j for skin in self.doc.get("skins", []) for j in skin["joints"]}
        return {"vertices": vertices, "faces": faces, "bone_count": len(joints), "meshes": meshes,
                "materials": self.doc.get("materials", []),
                "native_motion_names": [a.get("name", str(i)) for i, a in enumerate(self.doc.get("animations", []))],
                "scope": "RAW_BIND_MESH__DEFAULT_POSE_IS_NOT_BIND_ASSUMED"}
