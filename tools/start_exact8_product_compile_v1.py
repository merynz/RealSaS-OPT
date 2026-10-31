#!/usr/bin/env python3
from __future__ import annotations

"""Start a shipping product compile from exact controlled eight-view evidence.

This is an operator tool, not the public front/back UX. It requires the eight
observed RGBA PNGs plus the exact CameraProjectionBundle emitted by the
controlled source renderer. It never mirrors, synthesizes, recenters or guesses
missing views/cameras. The resulting SourceObservation.v1 enters the normal
product.input -> ... -> proof -> ProductRevision path.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

from compiler.realsas_compiler_services.platform_worker.source_observation_contract_v1 import (
    decode_source_observation_arrays_v1,
    encode_source_observation_arrays_v1,
)


VIEW_ORDER = ("S", "SE", "E", "NE", "N", "NW", "W", "SW")
VIEW_YAW_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
RESOLUTION = 1024
MIN_BORDER_MARGIN_PX = 8
CAMERA_TOL = 1e-5


class Exact8ProductError(RuntimeError):
    pass


def _finite_vector(raw: Any, *, name: str) -> np.ndarray:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise Exact8ProductError(f"EXACT8_CAMERA_VECTOR_INVALID:{name}")
    value = np.asarray(raw, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise Exact8ProductError(f"EXACT8_CAMERA_VECTOR_NONFINITE:{name}")
    return value


def _close(a: np.ndarray, b: np.ndarray, *, code: str, tolerance: float = CAMERA_TOL) -> None:
    if not np.allclose(a, b, rtol=0.0, atol=tolerance):
        raise Exact8ProductError(code)


def validate_controlled_camera_bundle_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    camera = dict(value)
    if camera.get("schema") != "RealSaS.CameraProjectionBundle.v1":
        raise Exact8ProductError("EXACT8_CAMERA_SCHEMA_INVALID")
    if tuple(map(str, camera.get("view_order") or ())) != VIEW_ORDER:
        raise Exact8ProductError("EXACT8_CAMERA_VIEW_ORDER_DRIFT")
    if camera.get("per_view_recenter", False) is not False:
        raise Exact8ProductError("EXACT8_CAMERA_PER_VIEW_RECENTER_FORBIDDEN")
    if camera.get("per_view_pose_change", False) is not False:
        raise Exact8ProductError("EXACT8_CAMERA_PER_VIEW_POSE_CHANGE_FORBIDDEN")

    center = _finite_vector(camera.get("common_center_xyz"), name="common_center_xyz")
    half = float(camera.get("common_half_extent", 0.0))
    if not math.isfinite(half) or half <= 0.0:
        raise Exact8ProductError("EXACT8_CAMERA_COMMON_HALF_EXTENT_INVALID")
    distance = max(4.0 * half, 1.0)

    rows = camera.get("cameras")
    if not isinstance(rows, list) or len(rows) != 8:
        raise Exact8ProductError("EXACT8_CAMERA_COUNT_INVALID")
    for index, (row, yaw_deg) in enumerate(zip(rows, VIEW_YAW_DEG)):
        if not isinstance(row, Mapping):
            raise Exact8ProductError(f"EXACT8_CAMERA_ROW_INVALID:V{index}")
        if str(row.get("schema_version") or "") != "RealSaS.FullSurfaceCameraProjection.v3":
            raise Exact8ProductError(f"EXACT8_CAMERA_PROJECTION_SCHEMA_DRIFT:V{index}")
        if str(row.get("view_id") or "") != f"V{index}" or int(row.get("view_index", -1)) != index:
            raise Exact8ProductError(f"EXACT8_CAMERA_IDENTITY_DRIFT:V{index}")
        if int(row.get("resolution", 0)) != RESOLUTION:
            raise Exact8ProductError(f"EXACT8_CAMERA_RESOLUTION_DRIFT:V{index}")
        row_half = float(row.get("half_extent", 0.0))
        if not math.isfinite(row_half) or abs(row_half - half) > CAMERA_TOL:
            raise Exact8ProductError(f"EXACT8_CAMERA_HALF_EXTENT_DRIFT:V{index}")

        theta = math.radians(yaw_deg)
        radial = np.asarray([math.sin(theta), -math.cos(theta), 0.0], dtype=np.float64)
        forward = -radial
        right = np.asarray([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        expected_origin = center + radial * distance

        _close(_finite_vector(row.get("origin"), name=f"V{index}.origin"), expected_origin,
               code=f"EXACT8_CAMERA_ORIGIN_DRIFT:V{index}")
        _close(_finite_vector(row.get("right"), name=f"V{index}.right"), right,
               code=f"EXACT8_CAMERA_RIGHT_DRIFT:V{index}")
        _close(_finite_vector(row.get("screen_up"), name=f"V{index}.screen_up"), up,
               code=f"EXACT8_CAMERA_UP_DRIFT:V{index}")
        _close(_finite_vector(row.get("forward"), name=f"V{index}.forward"), forward,
               code=f"EXACT8_CAMERA_FORWARD_DRIFT:V{index}")
    return camera


def load_controlled_camera_bundle_v1(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Exact8ProductError("EXACT8_CAMERA_JSON_INVALID") from exc
    if not isinstance(value, dict):
        raise Exact8ProductError("EXACT8_CAMERA_JSON_OBJECT_REQUIRED")
    return validate_controlled_camera_bundle_v1(value)


def _alpha_margin(alpha: np.ndarray, *, view_index: int) -> int:
    ys, xs = np.nonzero(alpha > 0)
    if not len(xs):
        raise Exact8ProductError(f"EXACT8_SOURCE_EMPTY_FOREGROUND:V{view_index}")
    height, width = alpha.shape
    margin = int(min(xs.min(), ys.min(), width - 1 - xs.max(), height - 1 - ys.max()))
    if margin < MIN_BORDER_MARGIN_PX:
        raise Exact8ProductError(
            f"EXACT8_SOURCE_BORDER_MARGIN_FAIL:V{view_index}:{margin}<{MIN_BORDER_MARGIN_PX}"
        )
    return margin


def load_exact8_rgba_v1(views_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    views_dir = views_dir.resolve()
    rows: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    expected_paths = [views_dir / f"V{index}.png" for index in range(8)]
    for index, path in enumerate(expected_paths):
        if not path.is_file():
            raise Exact8ProductError(f"EXACT8_SOURCE_VIEW_MISSING:V{index}")
        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise Exact8ProductError(f"EXACT8_SOURCE_FORMAT_INVALID:V{index}")
                if image.size != (RESOLUTION, RESOLUTION):
                    raise Exact8ProductError(f"EXACT8_SOURCE_RESOLUTION_DRIFT:V{index}")
                if "A" not in image.getbands():
                    raise Exact8ProductError(f"EXACT8_SOURCE_ALPHA_REQUIRED:V{index}")
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        except Exact8ProductError:
            raise
        except Exception as exc:
            raise Exact8ProductError(f"EXACT8_SOURCE_DECODE_FAILED:V{index}") from exc
        if rgba.shape != (RESOLUTION, RESOLUTION, 4):
            raise Exact8ProductError(f"EXACT8_SOURCE_RGBA_SHAPE_INVALID:V{index}")
        margin = _alpha_margin(rgba[..., 3], view_index=index)
        rows.append(np.moveaxis(rgba, -1, 0).copy())
        receipts.append({
            "view_index": index,
            "view_id": f"V{index}",
            "direction_label": VIEW_ORDER[index],
            "path": str(path),
            "foreground_pixel_count": int(np.count_nonzero(rgba[..., 3] > 0)),
            "border_margin_px": margin,
        })
    native_rgba = np.stack(rows, axis=0)[None, ...]
    if native_rgba.shape != (1, 8, 4, RESOLUTION, RESOLUTION):
        raise Exact8ProductError("EXACT8_SOURCE_STACK_SHAPE_INVALID")
    return native_rgba, receipts


def materialize_exact8_source_observation_v1(
    *, views_dir: Path, camera_bundle_path: Path,
) -> tuple[bytes, dict[str, Any]]:
    camera = load_controlled_camera_bundle_v1(camera_bundle_path.resolve())
    rgba, receipts = load_exact8_rgba_v1(views_dir.resolve())
    payload = encode_source_observation_arrays_v1(native_rgba=rgba, camera_bundle=camera)
    decoded = decode_source_observation_arrays_v1(payload)
    if not np.array_equal(decoded.native_rgba, rgba):
        raise Exact8ProductError("EXACT8_SOURCE_CODEC_RGBA_DRIFT")
    if decoded.camera_bundle != camera:
        raise Exact8ProductError("EXACT8_SOURCE_CODEC_CAMERA_DRIFT")
    return payload, {
        "schema_version": "RealSaS.Exact8ProductSourceReceipt.v1",
        "view_order": list(VIEW_ORDER),
        "views": receipts,
        "camera_contract": "CONTROLLED_8VIEW_SOURCE_REST_V2",
        "source_observation_schema": "RealSaS.SourceObservation.v1",
        "missing_views_synthesized": False,
        "camera_inferred": False,
        "per_view_recenter": False,
        "per_view_pose_change": False,
    }


def start_compile(
    *, api_root: str, operator_token: str, actor: str, display_name: str, payload: bytes,
) -> dict[str, Any]:
    root = api_root.strip().rstrip("/")
    parsed = urllib.parse.urlparse(root)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise Exact8ProductError("EXACT8_PLATFORM_URL_INVALID")
    if not operator_token.strip():
        raise Exact8ProductError("EXACT8_OPERATOR_TOKEN_REQUIRED")
    if not actor.strip():
        raise Exact8ProductError("EXACT8_OPERATOR_ACTOR_REQUIRED")
    request = urllib.request.Request(
        root + "/v1/operator/product/compiles/from-source-observation",
        data=payload,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/octet-stream",
            "Authorization": "Bearer " + operator_token.strip(),
            "X-RealSaS-Worker": actor.strip(),
            "X-RealSaS-Display-Name": display_name.strip() or "Exact 8-View Character",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read(2 * 1024 * 1024)
    except urllib.error.HTTPError as exc:
        detail = exc.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
        raise Exact8ProductError(f"EXACT8_PLATFORM_HTTP_{exc.code}:{detail}") from exc
    except urllib.error.URLError as exc:
        raise Exact8ProductError(f"EXACT8_PLATFORM_UNAVAILABLE:{exc.reason}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise Exact8ProductError("EXACT8_PLATFORM_RESPONSE_INVALID") from exc
    if not isinstance(value, dict) or not str(value.get("compile_id") or "").startswith("cmp_"):
        raise Exact8ProductError("EXACT8_PLATFORM_COMPILE_ID_INVALID")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views-dir", required=True, help="Directory containing exact V0.png ... V7.png")
    parser.add_argument("--camera-bundle", required=True, help="Exact controlled camera_bundle.json")
    parser.add_argument("--api-root", help="Platform API root. Omit with --output-only.")
    parser.add_argument("--display-name", default="Exact 8-View Character")
    parser.add_argument("--actor", default="exact8-product-launcher")
    parser.add_argument("--operator-token", default=os.environ.get("REALSAS_OPERATOR_TOKEN", ""))
    parser.add_argument("--receipt")
    parser.add_argument("--source-observation")
    parser.add_argument("--output-only", action="store_true", help="Validate/materialize without starting a compile")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload, receipt = materialize_exact8_source_observation_v1(
        views_dir=Path(args.views_dir), camera_bundle_path=Path(args.camera_bundle),
    )
    if args.source_observation:
        output = Path(args.source_observation).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        receipt["source_observation_path"] = str(output)
    if args.output_only:
        result: dict[str, Any] = {"status": "VALIDATED", "receipt": receipt}
    else:
        if not args.api_root:
            raise Exact8ProductError("EXACT8_PLATFORM_URL_REQUIRED")
        compile_result = start_compile(
            api_root=args.api_root,
            operator_token=args.operator_token,
            actor=args.actor,
            display_name=args.display_name,
            payload=payload,
        )
        result = {"status": "QUEUED", "compile": compile_result, "receipt": receipt}
    if args.receipt:
        receipt_path = Path(args.receipt).expanduser().resolve()
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
