from __future__ import annotations

"""Pinned FlexiCubes adapter for IRIS vNext.

The 21 learned values are not teacher labels.  They are the raw FlexiCubes
optimization variables: 12 beta edge weights, 8 alpha corner weights and one
gamma quad-split weight.  Geometry supervision reaches them only through the
differentiable extraction result.
"""

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import torch
import torch.nn.functional as F


FLEXICUBES_REPOSITORY = "https://github.com/nv-tlabs/FlexiCubes.git"
FLEXICUBES_COMMIT = "4cc7d6c3d0cee83c011ce36721b81adff0dd7db6"
FLEXICUBES_LICENSE = "Apache-2.0"
FLEXICUBES_PARAMETER_SEMANTICS = "BETA_EDGE_12__ALPHA_CORNER_8__GAMMA_SPLIT_1"


@dataclass(frozen=True)
class FlexiCubesParametersV1:
    beta_raw: torch.Tensor
    alpha_raw: torch.Tensor
    gamma_raw: torch.Tensor
    beta: torch.Tensor
    alpha: torch.Tensor
    gamma: torch.Tensor


def split_flexicubes_parameters_v1(
    raw: torch.Tensor,
    *,
    weight_scale: float = 0.99,
) -> FlexiCubesParametersV1:
    """Interpret the exact 12+8+1 FlexiCubes parameterization.

    The normalized values are diagnostic/explanatory outputs.  The pinned
    upstream operator receives the raw values and applies the same transforms
    internally, so RealSaS does not fork FlexiCubes numerical semantics.
    """
    if raw.ndim < 1 or raw.shape[-1] != 21:
        raise ValueError("VNEXT_FLEXICUBES_PARAMETER_SHAPE_INVALID")
    if not (0.0 < float(weight_scale) < 1.0):
        raise ValueError("VNEXT_FLEXICUBES_WEIGHT_SCALE_INVALID")
    beta_raw = raw[..., :12]
    alpha_raw = raw[..., 12:20]
    gamma_raw = raw[..., 20]
    beta = torch.tanh(beta_raw) * float(weight_scale) + 1.0
    alpha = torch.tanh(alpha_raw) * float(weight_scale) + 1.0
    gamma = torch.sigmoid(gamma_raw) * float(weight_scale) + (1.0 - float(weight_scale)) / 2.0
    return FlexiCubesParametersV1(
        beta_raw=beta_raw,
        alpha_raw=alpha_raw,
        gamma_raw=gamma_raw,
        beta=beta,
        alpha=alpha,
        gamma=gamma,
    )


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_pinned_flexicubes_root_v1(root: str | Path) -> dict[str, str]:
    root = Path(root).expanduser().resolve()
    if not (root / "flexicubes.py").is_file() or not (root / "tables.py").is_file():
        raise RuntimeError("VNEXT_FLEXICUBES_SOURCE_FILES_MISSING")
    try:
        head = _git(root, "rev-parse", "HEAD")
        remote = _git(root, "config", "--get", "remote.origin.url")
        dirty = _git(root, "status", "--porcelain", "--", "flexicubes.py", "tables.py")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("VNEXT_FLEXICUBES_GIT_IDENTITY_UNAVAILABLE") from exc
    if head != FLEXICUBES_COMMIT:
        raise RuntimeError(f"VNEXT_FLEXICUBES_COMMIT_DRIFT:{head}")
    normalized_remote = remote.removesuffix("/").removesuffix(".git")
    expected_remote = FLEXICUBES_REPOSITORY.removesuffix(".git")
    if normalized_remote != expected_remote:
        raise RuntimeError(f"VNEXT_FLEXICUBES_REMOTE_DRIFT:{remote}")
    if dirty:
        raise RuntimeError("VNEXT_FLEXICUBES_SOURCE_DIRTY")
    return {
        "repository": FLEXICUBES_REPOSITORY,
        "commit": FLEXICUBES_COMMIT,
        "license": FLEXICUBES_LICENSE,
        "parameter_semantics": FLEXICUBES_PARAMETER_SEMANTICS,
    }


def _load_source_module(root: Path) -> ModuleType:
    """Load the pinned two-file reference without contaminating global `tables`.

    The upstream reference uses `from tables import *`.  We temporarily bind its
    local `tables.py` to that import name while executing `flexicubes.py`, then
    restore any pre-existing module.  Imported table symbols are copied into the
    FlexiCubes module namespace, so restoring afterwards is safe.
    """
    old_tables = sys.modules.get("tables")
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(root))
        table_spec = importlib.util.spec_from_file_location(
            "tables", root / "tables.py"
        )
        if table_spec is None or table_spec.loader is None:
            raise RuntimeError("VNEXT_FLEXICUBES_TABLE_SPEC_INVALID")
        tables_module = importlib.util.module_from_spec(table_spec)
        sys.modules["tables"] = tables_module
        table_spec.loader.exec_module(tables_module)

        spec = importlib.util.spec_from_file_location(
            "realsas_pinned_flexicubes", root / "flexicubes.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("VNEXT_FLEXICUBES_MODULE_SPEC_INVALID")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        if old_tables is None:
            sys.modules.pop("tables", None)
        else:
            sys.modules["tables"] = old_tables


class PinnedFlexiCubesV1:
    """Exact upstream differentiable extractor behind a fail-closed binding."""

    def __init__(
        self,
        source_root: str | Path,
        *,
        device: torch.device | str,
        weight_scale: float = 0.99,
    ) -> None:
        self.source_root = Path(source_root).expanduser().resolve()
        self.identity = verify_pinned_flexicubes_root_v1(self.source_root)
        self.device = torch.device(device)
        self.weight_scale = float(weight_scale)
        module = _load_source_module(self.source_root)
        self.operator = module.FlexiCubes(
            device=str(self.device), weight_scale=self.weight_scale
        )

    def construct_iris_grid(self, resolution: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FlexiCubes grid in the IRIS normalized [-1,1]^3 domain."""
        if int(resolution) < 2:
            raise ValueError("VNEXT_FLEXICUBES_RESOLUTION_TOO_SMALL")
        vertices, cubes = self.operator.construct_voxel_grid(int(resolution))
        return vertices * 2.0, cubes.long()

    def extract(
        self,
        *,
        vertices_iris: torch.Tensor,
        field: torch.Tensor,
        cubes: torch.Tensor,
        raw_parameters: torch.Tensor,
        resolution: int,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parameters = split_flexicubes_parameters_v1(
            raw_parameters, weight_scale=self.weight_scale
        )
        if raw_parameters.shape != (cubes.shape[0], 21):
            raise ValueError("VNEXT_FLEXICUBES_CUBE_PARAMETER_COUNT_DRIFT")
        if field.shape != (vertices_iris.shape[0],):
            raise ValueError("VNEXT_FLEXICUBES_FIELD_SHAPE_INVALID")
        # Pass raw beta/alpha/gamma.  Upstream owns their normalization.
        vertices, faces, l_dev = self.operator(
            vertices_iris,
            field,
            cubes,
            int(resolution),
            beta_fx12=parameters.beta_raw,
            alpha_fx8=parameters.alpha_raw,
            gamma_f=parameters.gamma_raw,
            training=bool(training),
        )
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise RuntimeError("VNEXT_FLEXICUBES_VERTEX_OUTPUT_INVALID")
        if faces.ndim != 2 or faces.shape[-1] != 3:
            raise RuntimeError("VNEXT_FLEXICUBES_FACE_OUTPUT_INVALID")
        if vertices.numel() == 0 or faces.numel() == 0:
            raise RuntimeError("VNEXT_FLEXICUBES_EMPTY_SURFACE")
        return vertices, faces.long(), l_dev


def vertex_normals_v1(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    tri = vertices[faces]
    face_normal = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    out = torch.zeros_like(vertices)
    for corner in range(3):
        out = out.index_add(0, faces[:, corner], face_normal)
    return F.normalize(out, dim=-1, eps=1.0e-8)


def deterministic_surface_witness_v1(
    vertices_iris: torch.Tensor,
    faces: torch.Tensor,
    *,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(count) <= 0:
        raise ValueError("VNEXT_SURFACE_WITNESS_COUNT_INVALID")
    if vertices_iris.shape[0] <= 0:
        raise RuntimeError("VNEXT_SURFACE_WITNESS_EMPTY")
    normals = vertex_normals_v1(vertices_iris, faces)
    n = int(vertices_iris.shape[0])
    # Stable coverage of the extracted vertex stream.  If fewer vertices exist,
    # repeat deterministically rather than injecting teacher geometry.
    index = torch.floor(
        torch.arange(int(count), device=vertices_iris.device, dtype=torch.float64)
        * (float(n) / float(count))
    ).long().clamp_max(n - 1)
    return vertices_iris[index], normals[index]


def symmetric_surface_chamfer_v1(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    if predicted.ndim != 3 or teacher.ndim != 3 or predicted.shape[0] != teacher.shape[0]:
        raise ValueError("VNEXT_SURFACE_CHAMFER_BATCH_SHAPE_INVALID")
    distance = torch.cdist(predicted.float(), teacher.float(), p=2)
    return distance.min(dim=2).values.mean() + distance.min(dim=1).values.mean()


def flexicubes_root_from_environment_v1() -> Path:
    raw = os.environ.get("REALSAS_FLEXICUBES_ROOT", "").strip()
    if not raw:
        raise RuntimeError("VNEXT_FLEXICUBES_ROOT_REQUIRED")
    return Path(raw)
