from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
TYPE_WIDTH = {
    'SCALAR': 1,
    'VEC2': 2,
    'VEC3': 3,
    'VEC4': 4,
    'MAT2': 4,
    'MAT3': 9,
    'MAT4': 16,
}
ACTOR_ROLES = {'FULL_TRUTH_ACTOR_CANDIDATE', 'MIXED_STATIC_OR_ACTOR_AUDIT'}


def digest_json(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def round_float(value: float, digits: int = 7) -> float:
    v = round(float(value), digits)
    return 0.0 if v == 0.0 else v


def read_glb(path: Path) -> tuple[dict, list[bytes]]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b'glTF':
        raise ValueError('GLB_MAGIC')
    version, total = struct.unpack_from('<II', data, 4)
    if version != 2 or total != len(data):
        raise ValueError('GLB_HEADER')
    off = 12
    doc = None
    bins: list[bytes] = []
    while off + 8 <= len(data):
        length, ctype = struct.unpack_from('<II', data, off)
        off += 8
        payload = data[off:off + length]
        off += length
        if ctype == 0x4E4F534A:
            doc = json.loads(payload.rstrip(b'\x00 \t\r\n').decode('utf-8'))
        elif ctype == 0x004E4942:
            bins.append(payload)
    if doc is None:
        raise ValueError('GLB_JSON_MISSING')
    return doc, bins


def load_gltf(path: Path) -> tuple[dict, list[bytes]]:
    if path.suffix.lower() == '.glb':
        return read_glb(path)
    doc = json.loads(path.read_text(encoding='utf-8'))
    buffers: list[bytes] = []
    for row in doc.get('buffers') or []:
        uri = row.get('uri')
        if not uri:
            raise ValueError('GLTF_EXTERNAL_BUFFER_URI_MISSING')
        if uri.startswith('data:'):
            _, payload = uri.split(',', 1)
            buffers.append(base64.b64decode(payload))
        else:
            candidate = (path.parent / uri).resolve()
            if path.parent.resolve() not in candidate.parents and candidate != path.parent.resolve():
                raise ValueError('GLTF_BUFFER_PATH_ESCAPE')
            buffers.append(candidate.read_bytes())
    return doc, buffers


def accessor_array(doc: dict, buffers: list[bytes], index: int) -> np.ndarray:
    acc = doc['accessors'][index]
    if 'sparse' in acc:
        raise ValueError('SPARSE_ACCESSOR_UNSUPPORTED_FAIL_CLOSED')
    if 'bufferView' not in acc:
        raise ValueError('ACCESSOR_WITHOUT_BUFFERVIEW')
    view = doc['bufferViews'][acc['bufferView']]
    dtype = np.dtype(COMPONENT_DTYPE[acc['componentType']]).newbyteorder('<')
    width = TYPE_WIDTH[acc['type']]
    count = int(acc['count'])
    buffer_index = int(view.get('buffer', 0))
    raw = buffers[buffer_index]
    offset = int(view.get('byteOffset', 0)) + int(acc.get('byteOffset', 0))
    stride = int(view.get('byteStride', dtype.itemsize * width))
    packed = dtype.itemsize * width
    if stride == packed:
        arr = np.frombuffer(raw, dtype=dtype, count=count * width, offset=offset).reshape(count, width)
    else:
        arr = np.ndarray((count, width), dtype=dtype, buffer=raw, offset=offset,
                         strides=(stride, dtype.itemsize))
    return np.asarray(arr).copy()


def parent_index(nodes: list[dict]) -> list[int | None]:
    parents: list[int | None] = [None] * len(nodes)
    for i, node in enumerate(nodes):
        for child in node.get('children') or []:
            child = int(child)
            if parents[child] is not None and parents[child] != i:
                raise ValueError('NODE_MULTIPLE_PARENTS')
            parents[child] = i
    return parents


def node_local_signature(node: dict) -> dict:
    if 'matrix' in node:
        values = [round_float(v) for v in node['matrix']]
        return {'matrix': values}
    return {
        'translation': [round_float(v) for v in node.get('translation', [0, 0, 0])],
        'rotation': [round_float(v) for v in node.get('rotation', [0, 0, 0, 1])],
        'scale': [round_float(v) for v in node.get('scale', [1, 1, 1])],
    }


def skeleton_signature(doc: dict, skin_index: int) -> tuple[str, dict[str, int]]:
    nodes = list(doc.get('nodes') or [])
    parents = parent_index(nodes)
    skin = doc['skins'][skin_index]
    joint_indices = [int(x) for x in skin.get('joints') or []]
    if not joint_indices:
        raise ValueError('EMPTY_SKIN_JOINTS')
    names: dict[int, str] = {}
    for j in joint_indices:
        name = str(nodes[j].get('name') or '')
        if not name or name in names.values():
            raise ValueError('JOINT_NAME_MISSING_OR_DUPLICATE')
        names[j] = name
    joint_set = set(joint_indices)
    rows = []
    for j in joint_indices:
        p = parents[j]
        while p is not None and p not in joint_set:
            p = parents[p]
        rows.append({
            'name': names[j],
            'parent': names[p] if p is not None else None,
            'local': node_local_signature(nodes[j]),
        })
    rows.sort(key=lambda x: x['name'])
    return digest_json(rows), {name: i for i, name in names.items()}


def node_skin_for_mesh(doc: dict) -> dict[int, int]:
    result: dict[int, int] = {}
    for node in doc.get('nodes') or []:
        if 'mesh' not in node or 'skin' not in node:
            continue
        mesh = int(node['mesh'])
        skin = int(node['skin'])
        if mesh in result and result[mesh] != skin:
            raise ValueError('MESH_REFERENCED_BY_MULTIPLE_SKINS')
        result[mesh] = skin
    return result


def canonical_primitive_signature(doc: dict, buffers: list[bytes], mesh_index: int,
                                  primitive: dict, skin_index: int | None) -> dict:
    attrs = dict(primitive.get('attributes') or {})
    if 'POSITION' not in attrs:
        raise ValueError('PRIMITIVE_POSITION_REQUIRED')
    positions = accessor_array(doc, buffers, int(attrs['POSITION'])).astype(np.float64)
    positions = np.round(positions, 7)
    positions[positions == 0] = 0
    if 'indices' in primitive:
        indices = accessor_array(doc, buffers, int(primitive['indices'])).reshape(-1).astype(np.int64)
    else:
        indices = np.arange(len(positions), dtype=np.int64)
    if int(primitive.get('mode', 4)) != 4:
        topology = {'mode': int(primitive.get('mode', 4)), 'index_count': len(indices)}
    else:
        if len(indices) % 3:
            raise ValueError('TRIANGLE_INDEX_CARDINALITY')
        triangles = indices.reshape(-1, 3)
        # Preserve orientation but make triangle list independent of authored face order.
        canonical_faces = []
        for tri in triangles:
            tri = [int(x) for x in tri]
            rots = [tri, tri[1:] + tri[:1], tri[2:] + tri[:2]]
            canonical_faces.append(min(rots))
        canonical_faces.sort()
        topology = {'mode': 4, 'faces': canonical_faces}

    # Vertex order is still part of exact mechanical identity because skin rows address vertices.
    position_digest = hashlib.sha256(positions.astype('<f8').tobytes()).hexdigest()
    row = {
        'position_count': int(len(positions)),
        'position_digest': position_digest,
        'topology_digest': digest_json(topology),
        'skin_digest': None,
    }
    has_joints = 'JOINTS_0' in attrs
    has_weights = 'WEIGHTS_0' in attrs
    if has_joints != has_weights:
        raise ValueError('JOINT_WEIGHT_PAIR_REQUIRED')
    if has_joints:
        if skin_index is None:
            raise ValueError('SKIN_ATTRIBUTES_WITHOUT_SKIN_NODE')
        skin = doc['skins'][skin_index]
        joints = [int(x) for x in skin.get('joints') or []]
        nodes = list(doc.get('nodes') or [])
        names = [str(nodes[j].get('name') or '') for j in joints]
        if any(not x for x in names):
            raise ValueError('SKIN_JOINT_NAME_REQUIRED')
        ji = accessor_array(doc, buffers, int(attrs['JOINTS_0'])).astype(np.int64)
        wt = accessor_array(doc, buffers, int(attrs['WEIGHTS_0'])).astype(np.float64)
        if ji.shape != wt.shape or len(ji) != len(positions):
            raise ValueError('SKIN_ATTRIBUTE_SHAPE')
        influences = []
        for jrow, wrow in zip(ji, wt):
            pairs = []
            for j, w in zip(jrow, wrow):
                w = round_float(float(w), 7)
                if w == 0.0:
                    continue
                if int(j) < 0 or int(j) >= len(names):
                    raise ValueError('JOINT_INDEX_RANGE')
                pairs.append((names[int(j)], w))
            pairs.sort()
            influences.append(pairs)
        row['skin_digest'] = digest_json(influences)
    return row


def mechanical_identity(path: Path) -> dict:
    doc, buffers = load_gltf(path)
    skins = list(doc.get('skins') or [])
    mesh_skin = node_skin_for_mesh(doc)
    skeletons = []
    for i in range(len(skins)):
        signature, _ = skeleton_signature(doc, i)
        skeletons.append(signature)
    primitives = []
    for mi, mesh in enumerate(doc.get('meshes') or []):
        for primitive in mesh.get('primitives') or []:
            primitives.append(canonical_primitive_signature(doc, buffers, mi, primitive, mesh_skin.get(mi)))
    primitives.sort(key=lambda x: digest_json(x))
    actor = {
        'skeleton_signatures': sorted(skeletons),
        'primitives': primitives,
    }
    return {
        'mechanical_identity_sha256': digest_json(actor),
        'lineage_sha256': digest_json(sorted(skeletons)),
        'skin_count': len(skins),
        'animation_count': len(doc.get('animations') or []),
        'mesh_count': len(doc.get('meshes') or []),
        'primitive_count': len(primitives),
    }


def actor_candidate_paths(extracted: Path, role: str) -> list[Path]:
    if role not in ACTOR_ROLES:
        return []
    rows = []
    for path in sorted(extracted.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.gltf', '.glb'}:
            continue
        parts = {p.lower() for p in path.parts}
        if 'animations' in parts or 'assets' in parts:
            continue
        try:
            doc, _ = load_gltf(path)
        except Exception:
            continue
        if doc.get('skins'):
            rows.append(path)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', type=Path, required=True)
    ap.add_argument('--pack-id', required=True)
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    registry = json.loads(args.registry.read_text())
    pack = next((r for r in registry['packs'] if r['id'] == args.pack_id), None)
    if pack is None:
        raise SystemExit('KAYKIT_PACK_NOT_REGISTERED')
    audit_path = args.workspace / 'KAYKIT_FREE_SOURCE_AUDIT_V1.json'
    if not audit_path.is_file():
        raise SystemExit('KAYKIT_SOURCE_AUDIT_REQUIRED')
    source_audit = json.loads(audit_path.read_text())
    if source_audit.get('pack_id') != args.pack_id or source_audit.get('gltf_parse_failure_count') != 0:
        raise SystemExit('KAYKIT_SOURCE_AUDIT_NOT_CLEAN')
    extracted = args.workspace / 'raw' / '_extracted'
    if not extracted.is_dir():
        raise SystemExit('KAYKIT_EXTRACTED_SOURCE_REQUIRED')

    role = str(pack.get('expected_role') or '')
    candidates = actor_candidate_paths(extracted, role)
    rows = []
    failures = []
    for path in candidates:
        try:
            identity = mechanical_identity(path)
            rows.append({
                'relative_path': path.relative_to(extracted).as_posix(),
                **identity,
            })
        except Exception as exc:
            failures.append({
                'relative_path': path.relative_to(extracted).as_posix(),
                'error': type(exc).__name__ + ':' + str(exc),
            })

    mechanical_groups: dict[str, list[str]] = {}
    lineage_groups: dict[str, list[str]] = {}
    for row in rows:
        mechanical_groups.setdefault(row['mechanical_identity_sha256'], []).append(row['relative_path'])
        lineage_groups.setdefault(row['lineage_sha256'], []).append(row['relative_path'])

    declared = pack.get('declared_free_subject_count')
    receipt = {
        'schema': 'RealSaS.KayKitFree.ActorIdentityAudit.v1',
        'authority': 'IDENTITY_AND_LINEAGE_CENSUS__NOT_TRAINING_AUTHORITY',
        'pack_id': args.pack_id,
        'expected_role': role,
        'actor_export_candidate_count': len(rows),
        'exact_mechanical_identity_count': len(mechanical_groups),
        'leakage_lineage_count': len(lineage_groups),
        'declared_free_subject_count': declared,
        'declared_count_reconciled': declared is None or len(mechanical_groups) == int(declared),
        'mechanical_identity_groups': mechanical_groups,
        'leakage_lineage_groups': lineage_groups,
        'rows': rows,
        'failures': failures,
        'training_row_policy': 'EXACT_MECHANICAL_IDENTITIES_MAY_REMAIN_ROWS__LINEAGE_GROUPS_MUST_NEVER_CROSS_SPLITS',
        'subject_count_policy': 'DO_NOT_INFER_SEMANTIC_SUBJECT_COUNT_FROM_FILE_COUNT_OR_LINEAGE_COUNT',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print('KAYKIT_ACTOR_IDENTITY_CENSUS', json.dumps({
        'pack_id': args.pack_id,
        'actor_exports': len(rows),
        'mechanical_identities': len(mechanical_groups),
        'lineages': len(lineage_groups),
        'declared_subjects': declared,
        'declared_count_reconciled': receipt['declared_count_reconciled'],
        'failures': len(failures),
    }, sort_keys=True))
    if failures:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
