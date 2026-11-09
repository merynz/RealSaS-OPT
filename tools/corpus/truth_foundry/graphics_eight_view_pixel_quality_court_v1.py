#!/usr/bin/env python3
"""Examine real six-subject / eight-angle source-relative orbit raster evidence.

This is a *render-quality observation* court, not a source authoring/parity,
visual-art, anatomical-facing, or training admission court. Input scene+camera
receipts were produced by graphics_8view_source_teacher_pilot_v1.py.

Nondegenerate alpha and recorded camera/orbit must be tested; empty or
missing images fail closed. Palette/appearance consistency and symmetry are
reported as evidence/warnings, not passed as subjective art quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

AZIMUTHS = tuple(range(0, 360, 45))
CATEGORIES = ('serpentine', 'marine', 'insectoid', 'avian', 'quadrupedal', 'articulated_rigid')
SCHEMA = 'RealSaS.GraphicsEightViewPixelQualityCourt.v1'
SIZE = (256, 256)
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_GLB_BYTES = 32 * 1024 * 1024


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure(condition, label):
    if not condition:
        raise ValueError(label)


def parse_rows(working: Path, expect_subjects: int = 6):
    selection = working / 'SIX_SOURCE_SELECTED_MANIFEST_V1.csv'
    receipts = working / 'RENDER_RECEIPTS_V1.jsonl'
    ensure(selection.is_file(), 'SELECTION_MANIFEST_MISSING')
    ensure(receipts.is_file(), 'RENDER_RECEIPTS_MISSING')
    with selection.open('r', encoding='utf-8', newline='') as fp:
        sources = list(csv.DictReader(fp))
    ensure(len(sources) == expect_subjects, 'SELECTED_SOURCE_COUNT_MISMATCH')
    ensure(len({r['object_type'] for r in sources}) == len(sources), 'DUPLICATE_SELECTED_SOURCE')
    selected = {r['object_type']: r for r in sources}
    ensure(set(r['morphology_provider_claim'] for r in sources) <= set(CATEGORIES),
           'UNKNOWN_MORPHOLOGY_CLAIM')
    with receipts.open('r', encoding='utf-8') as fp:
        observations = [json.loads(line) for line in fp if line.strip()]
    ensure(len(observations) == expect_subjects * 8, 'EXPECTED_EIGHT_PER_SELECTED_SOURCE')
    for r in observations:
        uid = r.get('object_type')
        ensure(uid in selected, 'UNSELECTED_RENDER_SOURCE')
        ensure(r.get('glb_sha256') == selected[uid]['glb_sha256'], 'GLB_SOURCE_SHA_MISMATCH')
        ensure(r.get('training_authorized') is False, 'FALSE_TRAINING_AUTHORITY')
        ensure(r.get('camera_source_frame') == 'CANONICAL_GLB_BLENDER_IMPORT__ANATOMICAL_FRONT_UNKNOWN',
               'CAMERA_AUTHORITY_MISLABELED')
        ensure(r.get('source_pixel_original_authority') != 'VERIFIED', 'FALSE_PIXEL_ORIGIN_AUTHORITY')
        ensure(r.get('source_rights_court') == 'NOT_RUN', 'FALSE_RIGHTS_AUTHORITY')
        ensure(r.get('style_art_court') == 'PROCEDURAL_2D_CANDIDATE__NOT_APPROVED',
               'FALSE_ART_AUTHORITY')
    return sources, observations, selected, file_sha(selection), file_sha(receipts)


def relative_path(root: Path, value: str) -> Path:
    # Explicitly reject archive/path traversal and absolute source path manipulation.
    p = Path(value)
    ensure(not p.is_absolute() and '..' not in p.parts, 'UNSAFE_IMAGE_PATH')
    path = root / p
    ensure(path.suffix.lower() == '.png', 'NONPNG_IMAGE')
    ensure(path.exists() and path.is_file(), 'MISSING_IMAGE:' + str(p))
    ensure(0 < path.stat().st_size <= MAX_FILE_BYTES, 'IMAGE_FILE_SIZE_OUT_OF_BOUNDS:' + str(p))
    return path


def inspect_rgba(path: Path, image_mode='RGBA'):
    with Image.open(path) as img:
        ensure(img.size == SIZE, 'IMAGE_DIMENSION_MISMATCH:' + str(path))
        ensure(img.mode == image_mode, 'IMAGE_MODE_MISMATCH:' + str(path))
        img.load()
        rgba = img.convert('RGBA')
        alpha = rgba.getchannel('A')
        aextrema = alpha.getextrema()
        box = alpha.getbbox()
        ensure(box is not None and aextrema[1] > 0, 'EMPTY_ALPHA_SILHOUETTE:' + str(path))
        alpha_hist = alpha.histogram()
        coverage = 1.0 - alpha_hist[0] / (SIZE[0] * SIZE[1])
        full = alpha_hist[255] / (SIZE[0] * SIZE[1])
        col = ImageStat.Stat(rgba.convert('RGB')).mean
        return {'sha256': file_sha(path), 'alpha_coverage_fraction': round(coverage, 6),
                'alpha_opaque_fraction': round(full, 6),
                'alpha_bbox': list(box), 'mean_rgb': [round(x, 4) for x in col],
                'alpha': alpha.copy(), 'image': rgba.copy()}


def camera_signature(m):
    ensure(isinstance(m, list) and len(m) == 4 and all(isinstance(row, list) and len(row) == 4 for row in m),
           'CAMERA_MATRIX_NOT_4X4')
    nums = [v for row in m for v in row]
    ensure(all(isinstance(v, (float, int)) and math.isfinite(v) for v in nums),
           'CAMERA_MATRIX_INVALID')
    return tuple(round(float(v), 7) for v in nums)


def run(root: Path, expected_sources=6):
    root = Path(root)
    sources, records, selected, selection_sha, receipt_sha = parse_rows(root, expected_sources)
    groups = defaultdict(list)
    for r in records:
        groups[r['object_type']].append(r)
    out_views, out_subjects, warnings = [], [], []
    for source in sources:
        uid = source['object_type']
        rows = groups[uid]
        ensure(len(rows) == 8, 'SOURCE_MISSING_VIEW:' + uid)
        ensure(set(r['camera_azimuth_deg_relative'] for r in rows) == set(AZIMUTHS),
               'SOURCE_AZIMUTH_COVERAGE_MISMATCH:' + uid)
        ensure(all(isinstance(r['camera_azimuth_deg_relative'], int) for r in rows),
               'NONINTEGER_AZIMUTH')
        signatures = [camera_signature(r['camera_matrix_world_rowmajor']) for r in rows]
        ensure(len(set(signatures)) == 8, 'DEGENERATE_CAMERA_MATRICES:' + uid)
        by_angle = {r['camera_azimuth_deg_relative']: r for r in rows}
        native_rasters = {}
        orig_alpha_coverage = {}
        native_sha_set = set()
        for angle in AZIMUTHS:
            r = by_angle[angle]
            native = relative_path(root, r['native_rgba_path'])
            candidate = relative_path(root, r['style_candidate_path'])
            a = inspect_rgba(native)
            b = inspect_rgba(candidate)
            ensure(a['sha256'] == r['native_rgba_sha256'], 'NATIVE_PIXEL_SHA_MISMATCH:' + uid)
            ensure(b['sha256'] == r['style_candidate_sha256'], 'STYLE_PIXEL_SHA_MISMATCH:' + uid)
            ensure(ImageChops.difference(a['alpha'], b['alpha']).getbbox() is None,
                   'STYLIZATION_ALTERS_ALPHA:' + uid)
            # Full render still might not be an acceptable front or 2D art.
            native_rasters[angle] = a['image']
            orig_alpha_coverage[angle] = a['alpha_coverage_fraction']
            native_sha_set.add(a['sha256'])
            out_views.append({
                'source_key': 'UNIML3D:' + uid, 'weak_morphology_claim': source['morphology_provider_claim'],
                'canonical_glb_sha256': source['glb_sha256'], 'relative_angle_degrees': angle,
                'native_image_sha256': a['sha256'], 'candidate_image_sha256': b['sha256'],
                'native_alpha_coverage': a['alpha_coverage_fraction'],
                'native_alpha_opaque_fraction': a['alpha_opaque_fraction'],
                'native_alpha_bbox_json': json.dumps(a['alpha_bbox']),
                'candidate_same_source_alpha_verified': True,
                'camera_matrix_finite': True, 'anatomical_front_verified': False,
                'source_original_appearance_verified': False,
                'graphics_stylized_2d_art_pass': False,
                'teacher_student_1view_pair_admitted': False,
                'all_six_head_training_admitted': False,
            })
        if len(native_sha_set) < 2:
            warnings.append({'source_key': uid, 'warning': 'ALL_VIEWS_IDENTICAL_PIXEL_SHA'})
        diff_0_180 = ImageChops.difference(native_rasters[0], native_rasters[180]).convert('RGB')
        delta_0_180 = round(sum(ImageStat.Stat(diff_0_180).mean) / 3 / 255, 6)
        if delta_0_180 < 0.005:
            warnings.append({'source_key': uid, 'warning': 'FRONT_BACK_VIEW_PIXELS_NEAR_IDENTICAL'})
        if min(orig_alpha_coverage.values()) < 0.005:
            warnings.append({'source_key': uid, 'warning': 'NEAR_EMPTY_VISIBLE_SOURCE_SILHOUETTE'})
        if max(orig_alpha_coverage.values()) > 0.95:
            warnings.append({'source_key': uid, 'warning': 'SILHOUETTE_FILLS_ALMOST_ENTIRE_IMAGE'})
        out_subjects.append({
            'source_key': 'UNIML3D:' + uid,
            'weak_morphology_claim': source['morphology_provider_claim'],
            'canonical_glb_sha256': source['glb_sha256'],
            'orbit_relative_view_count': 8,
            'unique_native_view_sha_count': len(native_sha_set),
            'alpha_min_coverage': round(min(orig_alpha_coverage.values()), 6),
            'alpha_max_coverage': round(max(orig_alpha_coverage.values()), 6),
            'relative_0_vs_180_rgb_mean_absolute_difference': delta_0_180,
            'original_source_mesh_and_art_rights_verified': False,
            'actual_anatomical_front_verified': False,
            'eight_source_relative_views_observed': True,
            'source_derived_1view_back_pair_admitted': False,
            'commercial_ml_train_admitted': False,
        })
    def write_csv(name, rows):
        file = root / name
        with file.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return file_sha(file)
    vsum = write_csv('EIGHT_VIEW_PIXEL_ALPHA_CAMERA_EVIDENCE_V1.csv', out_views)
    ssum = write_csv('SOURCE_RELATIVE_ORBIT_8VIEW_QUALITY_V1.csv', out_subjects)
    summary = {
        'schema': SCHEMA, 'source_canonical_glbs_observed': len(out_subjects),
        'source_relative_view_receipts_validated': len(out_views),
        'alpha_same_native_vs_stylized_views': len(out_views),
        'camera_matrices_distinct_within_source': len(out_subjects),
        'nonempty_native_silhouette_views': len(out_views),
        'source_relative_pair_0_180_measured': len(out_subjects),
        'view_quality_warnings': warnings,
        'structural_render_receipts_qa_pass': True,
        'original_source_authority_pass': 0, 'anatomical_front_verified': 0,
        'approved_stylized_2d_art_targets': 0, 'teacher_back_pair_training_admitted': 0,
        'independent_training_eligible_six_head': 0,
        'training_authorized': False,
        'input_shas': {'selection': selection_sha, 'render_receipts': receipt_sha},
        'output_sha256': {'EIGHT_VIEW_PIXEL_ALPHA_CAMERA_EVIDENCE_V1.csv': vsum,
                          'SOURCE_RELATIVE_ORBIT_8VIEW_QUALITY_V1.csv': ssum}
    }
    (root / 'SOURCE_VIEW_QUALITY_SUMMARY_V1.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--expected-sources', type=int, default=6)
    args = p.parse_args()
    print(json.dumps(run(args.out, args.expected_sources), indent=2))

if __name__ == '__main__':
    main()