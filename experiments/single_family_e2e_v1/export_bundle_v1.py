from __future__ import annotations

import json
import re
from pathlib import Path

from compiler.realsas_compiler_core.bundle_routes import route_for
from compiler.realsas_compiler_core.v4 import require_current_proof_bundle, project_runtime_package_v3


def _write(root: Path, value):
    route = route_for(value)
    path = root / route.section / route.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return str(path.relative_to(root)).replace('\\', '/')


def _write_motion_bakes(root: Path, *, product, motion_bakes) -> list[str]:
    files = []
    seen = set()
    for bake in tuple(motion_bakes or ()):
        clip_id = str(getattr(bake, 'clip_id', '') or '')
        if not clip_id or clip_id in seen:
            raise ValueError('EXPORT_MOTION_BAKE_CLIP_ID_INVALID_OR_DUPLICATE')
        if str(getattr(bake, 'source_product_state_hash', '')) != str(product.product_state_hash):
            raise ValueError('EXPORT_MOTION_BAKE_STALE_PRODUCT_BINDING')
        seen.add(clip_id)
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', clip_id).strip('._') or 'clip'
        path = root / 'proof' / 'motion_bakes' / f'{safe}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bake.to_dict(), sort_keys=True, indent=2) + "\n", encoding='utf-8')
        files.append(str(path.relative_to(root)).replace('\\', '/'))
    return files


def export_product_bundle_v1(output_root, *, product, proof_bundle, motion_bakes=()):
    """Materialize current V4 product + proof-owned preview sidecars.

    Authority remains product + fresh PASS proof. Motion bakes are serialized only
    as exact proof-produced preview/export siblings; no solver is replayed here.
    """
    require_current_proof_bundle(product, proof_bundle, require_pass=True)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    files = []
    files.append(_write(root, product)); files.append(_write(root, proof_bundle)); files.append(_write(root, product.mechanical_state)); files.append(_write(root, product.directional_renderables)); files.append(_write(root, product.capability_contract)); files.append(_write(root, product.motion_state))
    direction_count = 0; component_count = 0
    for direction in product.directional_renderables.directions:
        files.append(_write(root, direction)); direction_count += 1
        for component in direction.components:
            files.append(_write(root, component)); files.append(_write(root, component.appearance)); component_count += 1
            for completion in component.completions: files.append(_write(root, completion))
    for track in product.motion_state.joint_tracks: files.append(_write(root, track))
    for track in product.motion_state.order_tracks: files.append(_write(root, track))
    for track in product.motion_state.visibility_tracks: files.append(_write(root, track))
    if direction_count != 8: raise ValueError('EXPORT_REQUIRES_EXACT_8_DIRECTIONS')
    motion_bake_files = _write_motion_bakes(root, product=product, motion_bakes=motion_bakes)
    files.extend(motion_bake_files)
    manifest = {'schema':'RealSaS.ExportBundle.v1','source_product_state_hash':product.product_state_hash,'source_proof_hash':proof_bundle.proof_bundle_hash,'direction_count':direction_count,'component_count':component_count,'motion_bake_files':motion_bake_files,'files':sorted(set(files)),'representation_class':product.representation_class,'full_3d_reconstruction_authority':False,'export_solver_replay_forbidden':True}
    runtime = project_runtime_package_v3(product, proof_bundle, manifest=manifest, runtime_payload_ref='bundle_manifest.json')
    files.append(_write(root, runtime)); manifest['runtime_package_file'] = route_for(runtime).section + '/' + route_for(runtime).filename
    manifest['files'] = sorted(set(files))
    (root / 'bundle_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding='utf-8')
    return runtime, manifest
