from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path

MODEL_SUFFIXES = {'.gltf', '.glb', '.fbx', '.obj'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_glb_json(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b'glTF':
        raise ValueError('GLB_MAGIC')
    version, total = struct.unpack_from('<II', data, 4)
    if version != 2 or total != len(data):
        raise ValueError('GLB_HEADER')
    off = 12
    while off + 8 <= len(data):
        length, ctype = struct.unpack_from('<II', data, off)
        off += 8
        payload = data[off:off + length]
        off += length
        if ctype == 0x4E4F534A:
            return json.loads(payload.rstrip(b'\x00 \t\r\n').decode('utf-8'))
    raise ValueError('GLB_JSON_MISSING')


def load_gltf_doc(path: Path) -> dict:
    if path.suffix.lower() == '.glb':
        return read_glb_json(path)
    return json.loads(path.read_text(encoding='utf-8'))


def gltf_stats(path: Path) -> dict:
    doc = load_gltf_doc(path)
    skins = list(doc.get('skins') or [])
    anims = list(doc.get('animations') or [])
    meshes = list(doc.get('meshes') or [])
    materials = list(doc.get('materials') or [])
    images = list(doc.get('images') or [])
    primitives = []
    weighted = 0
    primitive_count = 0
    for mesh in meshes:
        for primitive in mesh.get('primitives') or []:
            primitive_count += 1
            attrs = dict(primitive.get('attributes') or {})
            has_skin_attrs = 'JOINTS_0' in attrs and 'WEIGHTS_0' in attrs
            weighted += int(has_skin_attrs)
            primitives.append({
                'mode': int(primitive.get('mode', 4)),
                'has_indices': 'indices' in primitive,
                'has_position': 'POSITION' in attrs,
                'has_normal': 'NORMAL' in attrs,
                'has_texcoord0': 'TEXCOORD_0' in attrs,
                'has_joints0': 'JOINTS_0' in attrs,
                'has_weights0': 'WEIGHTS_0' in attrs,
                'material': primitive.get('material'),
            })
    channels = sum(len(a.get('channels') or []) for a in anims)
    samplers = sum(len(a.get('samplers') or []) for a in anims)
    joints = sum(len(s.get('joints') or []) for s in skins)
    return {
        'mesh_count': len(meshes),
        'primitive_count': primitive_count,
        'weighted_primitive_count': weighted,
        'skin_count': len(skins),
        'skin_joint_reference_count': joints,
        'animation_count': len(anims),
        'animation_channel_count': channels,
        'animation_sampler_count': samplers,
        'material_count': len(materials),
        'image_count': len(images),
        'node_count': len(doc.get('nodes') or []),
        'full_truth_actor_candidate': bool(skins and anims and weighted),
        'rigged_candidate': bool(skins and weighted),
        'animated_candidate': bool(anims),
        'appearance_address_candidate': bool(materials or images),
        'primitives': primitives,
    }


def safe_extract(archive: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        root = out.resolve()
        for info in z.infolist():
            target = (out / info.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError('ZIP_PATH_ESCAPE')
        z.extractall(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', type=Path, required=True)
    ap.add_argument('--pack-id', required=True)
    ap.add_argument('--download-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    registry = json.loads(args.registry.read_text())
    rows = {row['id']: row for row in registry['packs']}
    if args.pack_id not in rows:
        raise SystemExit('KAYKIT_PACK_NOT_REGISTERED')
    pack = rows[args.pack_id]
    root = args.download_root
    archives = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.zip')
    if not archives:
        raise SystemExit('KAYKIT_FREE_ARCHIVE_MISSING')
    if any(any(token in p.name.lower() for token in ('extra', 'source files', 'source_files', 'sourcefiles')) for p in archives):
        raise SystemExit('KAYKIT_PAID_TIER_ARCHIVE_FORBIDDEN')

    extracted = root / '_extracted'
    for i, archive in enumerate(archives):
        safe_extract(archive, extracted / f'{i:03d}_{archive.stem}')

    model_files = sorted(p for p in extracted.rglob('*') if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES)
    by_suffix: dict[str, int] = {}
    for p in model_files:
        by_suffix[p.suffix.lower()] = by_suffix.get(p.suffix.lower(), 0) + 1

    gltf_rows = []
    parse_failures = []
    for p in model_files:
        if p.suffix.lower() not in {'.gltf', '.glb'}:
            continue
        try:
            stats = gltf_stats(p)
            gltf_rows.append({
                'relative_path': p.relative_to(extracted).as_posix(),
                'sha256': sha256(p),
                **stats,
            })
        except Exception as exc:
            parse_failures.append({'relative_path': p.relative_to(extracted).as_posix(), 'error': type(exc).__name__ + ':' + str(exc)})

    full_truth = [r for r in gltf_rows if r['full_truth_actor_candidate']]
    rigged = [r for r in gltf_rows if r['rigged_candidate']]
    animated = [r for r in gltf_rows if r['animated_candidate']]
    static = [r for r in gltf_rows if not r['rigged_candidate'] and not r['animated_candidate']]

    archive_rows = [{
        'name': p.name,
        'size_bytes': p.stat().st_size,
        'sha256': sha256(p),
    } for p in archives]

    receipt = {
        'schema': 'RealSaS.KayKitFree.SourceAudit.v1',
        'authority': 'STRUCTURAL_CENSUS__NOT_TRAINING_AUTHORITY',
        'pack_id': args.pack_id,
        'pack_name': pack['name'],
        'official_page': pack['official_page'],
        'license': registry['license'],
        'free_tier_only': True,
        'archives': archive_rows,
        'model_file_count': len(model_files),
        'model_format_counts': by_suffix,
        'gltf_or_glb_count': len(gltf_rows),
        'gltf_parse_failure_count': len(parse_failures),
        'full_truth_actor_candidate_count': len(full_truth),
        'rigged_candidate_count': len(rigged),
        'animated_candidate_count': len(animated),
        'static_gltf_candidate_count': len(static),
        'animation_clip_count': sum(r['animation_count'] for r in gltf_rows),
        'animation_channel_count': sum(r['animation_channel_count'] for r in gltf_rows),
        'skin_count': sum(r['skin_count'] for r in gltf_rows),
        'weighted_primitive_count': sum(r['weighted_primitive_count'] for r in gltf_rows),
        'parse_failures': parse_failures,
        'gltf_rows': gltf_rows,
        'admission': 'AUDIT_PENDING',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print('KAYKIT_FREE_STRUCTURAL_CENSUS', json.dumps({k: receipt[k] for k in (
        'pack_id', 'model_file_count', 'gltf_or_glb_count', 'full_truth_actor_candidate_count',
        'rigged_candidate_count', 'animated_candidate_count', 'static_gltf_candidate_count',
        'animation_clip_count', 'gltf_parse_failure_count')}, sort_keys=True))
    if parse_failures:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
