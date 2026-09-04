from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tga', '.tif', '.tiff', '.webp', '.dds', '.exr', '.hdr'}
_GENERIC_SOURCE_TOKENS = {'anim', 'animation', 'animated', 'model', 'mesh', 'rig', 'fbx', 'glb', 'gltf',
                          'character', 'char', 'texture', 'tex', 'material', 'mat', 'asset'}
NATIVE_RESOLUTION = 1024
VIEWS = 8


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def _norm_material_name(name):
    return re.sub(r'\.\d{3}$', '', (name or '').strip().lower())


def _name_tokens(s):
    return [t for t in re.split(r'[^a-z0-9]+', (s or '').lower())
            if t and t not in _GENERIC_SOURCE_TOKENS and len(t) >= 3]


def unity_guid_to_asset(scratch: Path):
    out = {}
    for meta in scratch.rglob('*.meta'):
        try:
            txt = meta.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        m = re.search(r'(?m)^guid:\s*([0-9a-fA-F]{32})\s*$', txt)
        if not m:
            continue
        asset = Path(str(meta)[:-5])
        if asset.exists() and asset.is_file():
            out[m.group(1).lower()] = asset
    return out


def unity_material_textures(scratch: Path):
    guid_map = unity_guid_to_asset(scratch)
    out, evidence = {}, {}
    for mat in scratch.rglob('*.mat'):
        try:
            txt = mat.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        pos = txt.find('Material:')
        body = txt[pos:] if pos >= 0 else txt
        mn = re.search(r'(?m)^\s*m_Name:\s*(.*?)\s*$', body)
        if not mn or not mn.group(1).strip():
            continue
        name = _norm_material_name(mn.group(1))
        guid = None
        for prop in ('_BaseMap', '_MainTex'):
            mm = re.search(rf'-\s*{re.escape(prop)}:\s*\n\s*m_Texture:\s*\{{[^}}]*guid:\s*([0-9a-fA-F]{{32}})', body)
            if mm:
                guid = mm.group(1).lower()
                break
        if guid and guid in guid_map and guid_map[guid].suffix.lower() in IMAGE_EXTS:
            out[name] = guid_map[guid]
            evidence[name] = {
                'material_file': str(mat.relative_to(scratch)),
                'texture_file': str(guid_map[guid].relative_to(scratch)),
                'guid': guid,
            }
    return out, evidence


def source_scoped_unity_material(source_file: Path, unity_map):
    src_tokens = set(_name_tokens(source_file.stem))
    candidates = []
    for mat_name, path in unity_map.items():
        overlap = sorted(src_tokens & set(_name_tokens(mat_name)))
        if overlap:
            candidates.append({'material_name': mat_name, 'texture_path': str(path), 'overlap': overlap})
    ev = {'source_stem': source_file.stem, 'source_tokens': sorted(src_tokens), 'candidates': candidates}
    return (Path(candidates[0]['texture_path']) if len(candidates) == 1 else None), ev


def resolve_material_image(material: dict, scratch: Path, unity_map):
    ip = material.get('image_path')
    if ip and Path(ip).exists():
        return Path(ip), 'BLENDER_DIRECT'
    name = material.get('image_name')
    if name:
        basename = Path(name).name.lower()
        hits = [p for p in scratch.rglob('*') if p.is_file() and p.name.lower() == basename]
        if len(hits) == 1:
            return hits[0], 'BLENDER_IMAGE_NAME'
        stem = Path(name).stem.lower()
        hits = [p for p in scratch.rglob('*')
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem.lower() == stem]
        if len(hits) == 1:
            return hits[0], 'BLENDER_IMAGE_STEM'
    mn = _norm_material_name(material.get('name'))
    if mn in unity_map:
        return unity_map[mn], 'UNITY_MATERIAL_GUID'
    if len(unity_map) == 1:
        return next(iter(unity_map.values())), 'UNITY_UNIQUE_MATERIAL_FALLBACK'
    return None, None


def load_texture(path: Path):
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert('RGBA'), dtype=np.float32) / 255.0
    except Exception:
        return None


def bilinear_sample(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = tex.shape[:2]
    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(uv[:, 1], 1.0)
    x = u * (w - 1)
    y = (1 - v) * (h - 1)
    x0 = np.floor(x).astype(np.int64).clip(0, w - 1)
    y0 = np.floor(y).astype(np.int64).clip(0, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    c0 = tex[y0, x0] * (1 - wx) + tex[y0, x1] * wx
    c1 = tex[y1, x0] * (1 - wx) + tex[y1, x1] * wx
    return c0 * (1 - wy) + c1 * wy


def build_material_runtime(source_file: Path, scratch: Path, structure: dict):
    unity_map, unity_evidence = unity_material_textures(scratch)
    direct = [resolve_material_image(m, scratch, unity_map) for m in structure['materials']]
    scoped_img, scoped_evidence = source_scoped_unity_material(source_file, unity_map)
    allow_scoped = len(structure['materials']) == 1 and scoped_img is not None
    mats, resolved = [], []
    for m, (img, method) in zip(structure['materials'], direct):
        if img is None and allow_scoped:
            img = scoped_img
            method = 'UNITY_SOURCE_TOKEN_UNIQUE_SINGLE_IMPORTED_MATERIAL'
        mats.append({
            'base_color': np.asarray(m.get('base_color', [.8, .8, .8, 1])[:4], dtype=np.float32),
            'texture': load_texture(img) if img else None,
            'image_path': str(img) if img else None,
            'resolution_method': method,
        })
        resolved.append({
            'imported_material_name': m.get('name'),
            'image_path': str(img) if img else None,
            'resolution_method': method,
        })
    return mats, {
        'unity_material_evidence': unity_evidence,
        'source_scoped_fallback_evidence': scoped_evidence,
        'effective_material_channels': len(structure['materials']),
        'source_scoped_fallback_allowed': bool(allow_scoped),
        'resolved_materials': resolved,
    }


def _qpoint(p: np.ndarray, tol: float) -> tuple[int, int, int]:
    return tuple(np.rint(np.asarray(p, np.float64) / tol).astype(np.int64).tolist())


def _triangle_key(points: np.ndarray, tol: float):
    return tuple(sorted(_qpoint(p, tol) for p in points))


def _corner_permutations(cpts: np.ndarray, epts: np.ndarray, tol: float):
    good = []
    for perm in itertools.permutations(range(3)):
        err = float(np.max(np.abs(cpts - epts[list(perm)])))
        if err <= tol:
            good.append((perm, err))
    return good


def align_appearance_to_canonical(
    canonical_geometry: Path,
    appearance_npz: Path,
    tol: float = 2e-6,
):
    """Return appearance arrays in canonical face/corner order.

    Exact index identity is preferred. If a Blender importer changes only mesh/object
    enumeration or vertex/face indices, a fail-closed geometric triangle bijection is
    allowed. Coordinate transforms, missing/extra triangles, ambiguous duplicate faces,
    or topology changes remain hard failures.
    """
    with np.load(canonical_geometry, allow_pickle=False) as c, np.load(appearance_npz, allow_pickle=False) as e:
        cv = np.asarray(c['vertices_source'], np.float32)
        cf = np.asarray(c['faces'], np.int32)
        ev = np.asarray(e['vertices_source'], np.float32)
        ef = np.asarray(e['faces'], np.int32)
        euv = np.asarray(e['face_uv'], np.float32)
        euv_valid = np.asarray(e['face_uv_valid'], np.uint8).astype(bool)
        emat = np.asarray(e['face_material'], np.int32)

    if cv.shape != ev.shape or cf.shape != ef.shape:
        raise RuntimeError(f'SOURCE_CANONICAL_SHAPE_MISMATCH:{ev.shape}/{cv.shape}:{ef.shape}/{cf.shape}')
    if euv.shape != (len(ef), 3, 2) or euv_valid.shape != (len(ef),) or emat.shape != (len(ef),):
        raise RuntimeError('SOURCE_APPEARANCE_PAYLOAD_SHAPE_DRIFT')

    direct_err = float(np.max(np.abs(cv - ev))) if cv.size else 0.0
    direct_faces_equal = bool(np.array_equal(cf, ef))
    if direct_err <= tol and direct_faces_equal:
        return euv, euv_valid, emat, {
            'mode': 'EXACT_INDEX_IDENTITY',
            'vertex_index_max_abs_err': direct_err,
            'faces_index_equal': True,
            'face_bijection_count': int(len(cf)),
            'face_bijection_max_abs_err': direct_err,
            'ambiguous_duplicate_triangle_buckets': 0,
        }

    extracted_buckets: dict[tuple, list[int]] = {}
    for fi, f in enumerate(ef):
        key = _triangle_key(ev[f], tol)
        extracted_buckets.setdefault(key, []).append(fi)
    canonical_buckets: dict[tuple, list[int]] = {}
    for fi, f in enumerate(cf):
        key = _triangle_key(cv[f], tol)
        canonical_buckets.setdefault(key, []).append(fi)

    if set(extracted_buckets) != set(canonical_buckets):
        missing = len(set(canonical_buckets) - set(extracted_buckets))
        extra = len(set(extracted_buckets) - set(canonical_buckets))
        raise RuntimeError(
            f'SOURCE_CANONICAL_GEOMETRY_DRIFT:no_triangle_key_bijection:missing_keys={missing}:extra_keys={extra}:'
            f'direct_vertex_max_abs={direct_err}:faces_equal={direct_faces_equal}'
        )

    aligned_uv = np.empty_like(euv)
    aligned_valid = np.empty_like(euv_valid)
    aligned_mat = np.empty_like(emat)
    max_err = 0.0
    ambiguous_buckets = 0

    for key, cfaces in canonical_buckets.items():
        efaces = extracted_buckets[key]
        if len(cfaces) != len(efaces):
            raise RuntimeError(f'SOURCE_CANONICAL_TOPOLOGY_MULTIPLICITY_DRIFT:key={key}:canonical={len(cfaces)}:extracted={len(efaces)}')
        if len(cfaces) > 1:
            ambiguous_buckets += 1
            raise RuntimeError(
                f'SOURCE_CANONICAL_AMBIGUOUS_DUPLICATE_TRIANGLE_BUCKET:key={key}:count={len(cfaces)}'
            )
        ci, ei = cfaces[0], efaces[0]
        cpts = cv[cf[ci]]
        epts = ev[ef[ei]]
        perms = _corner_permutations(cpts, epts, tol)
        if not perms:
            raise RuntimeError(f'SOURCE_CANONICAL_CORNER_BIJECTION_FAIL:canonical_face={ci}:extracted_face={ei}')
        unique_perms = {p for p, _ in perms}
        if len(unique_perms) != 1:
            raise RuntimeError(f'SOURCE_CANONICAL_AMBIGUOUS_CORNER_BIJECTION:canonical_face={ci}:extracted_face={ei}')
        perm, err = perms[0]
        max_err = max(max_err, err)
        aligned_uv[ci] = euv[ei][list(perm)]
        aligned_valid[ci] = euv_valid[ei]
        aligned_mat[ci] = emat[ei]

    return aligned_uv, aligned_valid, aligned_mat, {
        'mode': 'GEOMETRIC_FACE_BIJECTION',
        'vertex_index_max_abs_err': direct_err,
        'faces_index_equal': direct_faces_equal,
        'face_bijection_count': int(len(cf)),
        'face_bijection_max_abs_err': float(max_err),
        'ambiguous_duplicate_triangle_buckets': int(ambiguous_buckets),
    }


def render_asset(
    *,
    asset_id: str,
    candidate_id: str,
    source_sha256: str,
    source_file: Path,
    source_package: Path,
    canonical_geometry: Path,
    appearance_npz: Path,
    appearance_json: Path,
    canonical_render_root: Path,
    output_root: Path,
    min_texture_coverage: float = .05,
):
    face_uv, face_uv_valid, face_mat, geometry_alignment = align_appearance_to_canonical(
        canonical_geometry, appearance_npz
    )
    with np.load(canonical_geometry, allow_pickle=False) as c:
        faces = np.asarray(c['faces'], np.int32)
    structure = json.loads(appearance_json.read_text(encoding='utf-8'))
    mats, material_resolution = build_material_runtime(source_file, source_package, structure)
    used = sorted(set(int(x) for x in np.unique(face_mat).tolist()))
    bad = [mi for mi in used if mi < 0 or mi >= len(mats)]
    if bad:
        raise RuntimeError(f'USED_MATERIAL_INDEX_INVALID:{bad}')
    if not any(m['texture'] is not None for m in mats):
        raise RuntimeError('NO_SOURCE_TEXTURE_RESOLVED')

    output_root.mkdir(parents=True, exist_ok=True)
    views = []
    total_fg = total_tex = 0
    for vi in range(VIEWS):
        canon = canonical_render_root / f'V{vi}'
        ra_path = canon / 'raster_authority.npz'
        cam_path = canon / 'camera.json'
        if not ra_path.is_file() or not cam_path.is_file():
            raise RuntimeError(f'CANONICAL_VIEW_AUTHORITY_MISSING:V{vi}')
        with np.load(ra_path, allow_pickle=False) as ra:
            pix = np.asarray(ra['pixel_linear_index'], np.int64)
            tids = np.asarray(ra['triangle_id'], np.int64)
            buv = np.asarray(ra['barycentric_uv'], np.float32)
            res = np.asarray(ra['resolution'], np.int64).tolist()
            origin = str(np.asarray(ra['origin']).reshape(-1)[0])
        if res != [NATIVE_RESOLUTION, NATIVE_RESOLUTION] or origin != 'TOP_LEFT':
            raise RuntimeError(f'RASTER_CONTRACT_DRIFT:V{vi}:{res}:{origin}')
        if len(pix) != len(tids) or len(pix) != len(buv) or len(np.unique(pix)) != len(pix):
            raise RuntimeError(f'RASTER_CARDINALITY_DRIFT:V{vi}')
        if len(tids) and (tids.min() < 0 or tids.max() >= len(faces)):
            raise RuntimeError(f'TRIANGLE_ID_OUT_OF_RANGE:V{vi}')

        w = np.stack([buv[:, 0], buv[:, 1], 1 - buv[:, 0] - buv[:, 1]], axis=1)
        surface_uv = (face_uv[tids] * w[:, :, None]).sum(1)
        mat_ids = face_mat[tids]
        uv_ok = face_uv_valid[tids]
        pix_rgb = np.zeros((len(pix), 3), np.float32)
        textured = 0
        for mi, m in enumerate(mats):
            sel = mat_ids == mi
            if not np.any(sel):
                continue
            pix_rgb[sel] = m['base_color'][:3]
            if m['texture'] is not None:
                st = sel & uv_ok
                if np.any(st):
                    pix_rgb[st] = bilinear_sample(m['texture'], surface_uv[st])[:, :3]
                    textured += int(st.sum())

        rgba = np.zeros((NATIVE_RESOLUTION, NATIVE_RESOLUTION, 4), np.uint8)
        ys = pix // NATIVE_RESOLUTION
        xs = pix % NATIVE_RESOLUTION
        rgba[ys, xs, :3] = (np.clip(pix_rgb, 0, 1) * 255 + .5).astype(np.uint8)
        rgba[ys, xs, 3] = 255
        out_v = output_root / f'V{vi}'
        out_v.mkdir(parents=True, exist_ok=True)
        rgba_path = out_v / 'RGBA.png'
        Image.fromarray(rgba, 'RGBA').save(rgba_path, compress_level=6)
        shutil.copy2(cam_path, out_v / 'camera.json')
        total_fg += len(pix)
        total_tex += textured
        views.append({
            'view_index': vi,
            'yaw_deg': vi * 45,
            'rgba_path': str(rgba_path),
            'rgba_sha256': sha256_file(rgba_path),
            'camera_path': str(out_v / 'camera.json'),
            'camera_sha256': sha256_file(out_v / 'camera.json'),
            'canonical_raster_authority_sha256': sha256_file(ra_path),
            'foreground_pixels': len(pix),
            'textured_foreground_pixels': textured,
            'textured_foreground_fraction': float(textured / max(len(pix), 1)),
        })

    coverage = float(total_tex / max(total_fg, 1))
    if coverage < min_texture_coverage:
        raise RuntimeError(f'SOURCE_TEXTURE_COVERAGE_TOO_LOW:{coverage:.6f}<{min_texture_coverage:.6f}')

    manifest = {
        'schema': 'RealSaS.PrefitObservationManifest.v1',
        'asset_id': asset_id,
        'candidate_id': candidate_id,
        'source_sha256': source_sha256,
        'raster_authority': 'MASTER_SOURCE_TEXTURED_RGBA',
        'native_resolution': NATIVE_RESOLUTION,
        'exact_eight_views': True,
        'source_geometry_alignment': geometry_alignment,
        'textured_foreground_fraction': coverage,
        'material_resolution': material_resolution,
        'views': views,
        'render_geometry_authority': 'EXISTING_MASTER_RASTER_AUTHORITY',
        'camera_authority': 'EXISTING_MASTER_CAMERA_JSON',
        'rerasterized': False,
    }
    (output_root / 'PREFIT_OBSERVATION_MANIFEST_V1.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    return manifest


def main():
    ap = argparse.ArgumentParser()
    for x in (
        'asset-id', 'candidate-id', 'source-sha256', 'source-file', 'source-package',
        'canonical-geometry', 'appearance-npz', 'appearance-json', 'canonical-render-root',
        'output-root',
    ):
        ap.add_argument('--' + x, required=True)
    ap.add_argument('--min-texture-coverage', type=float, default=.05)
    a = ap.parse_args()
    print(json.dumps({
        'status': 'PASS',
        'manifest': render_asset(
            asset_id=a.asset_id,
            candidate_id=a.candidate_id,
            source_sha256=a.source_sha256,
            source_file=Path(a.source_file),
            source_package=Path(a.source_package),
            canonical_geometry=Path(a.canonical_geometry),
            appearance_npz=Path(a.appearance_npz),
            appearance_json=Path(a.appearance_json),
            canonical_render_root=Path(a.canonical_render_root),
            output_root=Path(a.output_root),
            min_texture_coverage=a.min_texture_coverage,
        ),
    }, sort_keys=True))


if __name__ == '__main__':
    main()
