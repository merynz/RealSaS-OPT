from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence

try:
    import numpy as np
    from skimage.draw import polygon as _raster_polygon
except ImportError:  # pragma: no cover - deterministic scalar fallback
    np = None
    _raster_polygon = None

Point2 = tuple[float, float]
Triangle2 = tuple[Point2, Point2, Point2]


def _mask_bytes(rows: Sequence[Sequence[object]]) -> tuple[int, int, bytes]:
    height = len(rows)
    if height <= 0:
        raise ValueError("observation mask must have at least one row")
    width = len(rows[0])
    if width <= 0:
        raise ValueError("observation mask must have at least one column")
    packed = bytearray()
    for row in rows:
        if len(row) != width:
            raise ValueError("observation mask rows must have equal width")
        packed.extend(1 if bool(v) else 0 for v in row)
    return width, height, bytes(packed)


def _signed_area2(a: Point2, b: Point2, c: Point2) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _point_in_triangle(p: Point2, tri: Triangle2, *, eps: float = 1.0e-9) -> bool:
    a, b, c = tri
    s0 = _signed_area2(a, b, p)
    s1 = _signed_area2(b, c, p)
    s2 = _signed_area2(c, a, p)
    has_neg = s0 < -eps or s1 < -eps or s2 < -eps
    has_pos = s0 > eps or s1 > eps or s2 > eps
    return not (has_neg and has_pos)


def _scalar_triangle_pixels(tri: Triangle2, *, width: int, height: int) -> set[tuple[int, int]]:
    xs = [float(p[0]) for p in tri]
    ys = [float(p[1]) for p in tri]
    xmin = max(0, int(math.ceil(min(xs) - 1.0e-9)))
    xmax = min(width - 1, int(math.floor(max(xs) + 1.0e-9)))
    ymin = max(0, int(math.ceil(min(ys) - 1.0e-9)))
    ymax = min(height - 1, int(math.floor(max(ys) + 1.0e-9)))
    if xmin > xmax or ymin > ymax:
        return set()
    out: set[tuple[int, int]] = set()
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            if _point_in_triangle((float(x), float(y)), tri):
                out.add((x, y))
    return out


def _triangle_indices(tri: Triangle2, *, width: int, height: int):
    if _raster_polygon is None:
        return None
    return _raster_polygon(
        [float(p[1]) for p in tri],
        [float(p[0]) for p in tri],
        shape=(int(height), int(width)),
    )


def _connected_component_metrics(
    authority_bytes: bytes,
    predicted_bytes: bytes,
    *,
    width: int,
    height: int,
    max_component_rows: int = 64,
) -> dict:
    """Exact 4-connected alpha accounting, including large-hole visibility.

    Aggregate recall alone can hide a missing hat/book/limb. These summaries make
    connected foreground loss explicit without assigning semantic component names.
    """
    n = int(width) * int(height)
    if len(authority_bytes) != n or len(predicted_bytes) != n:
        raise ValueError("coverage component mask byte count mismatch")

    authority = authority_bytes
    predicted = predicted_bytes
    visited = bytearray(n)
    components: list[tuple[int, int]] = []  # (foreground pixels, covered pixels)

    for seed in range(n):
        if not authority[seed] or visited[seed]:
            continue
        visited[seed] = 1
        q = deque([seed])
        size = covered = 0
        while q:
            idx = q.popleft()
            size += 1
            covered += 1 if predicted[idx] else 0
            x = idx % width
            y = idx // width
            if x > 0:
                nxt = idx - 1
                if authority[nxt] and not visited[nxt]: visited[nxt] = 1; q.append(nxt)
            if x + 1 < width:
                nxt = idx + 1
                if authority[nxt] and not visited[nxt]: visited[nxt] = 1; q.append(nxt)
            if y > 0:
                nxt = idx - width
                if authority[nxt] and not visited[nxt]: visited[nxt] = 1; q.append(nxt)
            if y + 1 < height:
                nxt = idx + width
                if authority[nxt] and not visited[nxt]: visited[nxt] = 1; q.append(nxt)
        components.append((size, covered))

    foreground = int(sum(authority))
    components.sort(key=lambda row: (-row[0], -row[1]))
    component_rows = tuple(
        {
            "component_index": int(i),
            "foreground_pixel_count": int(size),
            "covered_pixel_count": int(covered),
            "foreground_fraction": 0.0 if foreground == 0 else float(size) / float(foreground),
            "recall": 1.0 if size == 0 else float(covered) / float(size),
        }
        for i, (size, covered) in enumerate(components[: int(max_component_rows)])
    )

    uncovered = bytearray(n)
    for i in range(n):
        uncovered[i] = 1 if authority[i] and not predicted[i] else 0
    visited = bytearray(n)
    uncovered_sizes: list[int] = []
    for seed in range(n):
        if not uncovered[seed] or visited[seed]:
            continue
        visited[seed] = 1
        q = deque([seed])
        size = 0
        while q:
            idx = q.popleft(); size += 1
            x = idx % width; y = idx // width
            if x > 0:
                nxt = idx - 1
                if uncovered[nxt] and not visited[nxt]: visited[nxt] = 1; q.append(nxt)
            if x + 1 < width:
                nxt = idx + 1
                if uncovered[nxt] and not visited[nxt]: visited[nxt] = 1; q.append(nxt)
            if y > 0:
                nxt = idx - width
                if uncovered[nxt] and not visited[nxt]: visited[nxt] = 1; q.append(nxt)
            if y + 1 < height:
                nxt = idx + width
                if uncovered[nxt] and not visited[nxt]: visited[nxt] = 1; q.append(nxt)
        uncovered_sizes.append(size)

    largest_uncovered = max(uncovered_sizes, default=0)
    return {
        "foreground_component_count": len(components),
        "foreground_component_recalls": component_rows,
        "uncovered_component_count": len(uncovered_sizes),
        "largest_uncovered_component_pixels": int(largest_uncovered),
        "largest_uncovered_component_fraction": (
            0.0 if foreground == 0 else float(largest_uncovered) / float(foreground)
        ),
    }


@dataclass(frozen=True)
class ObservationRasterDomain:
    """Exact row-major alpha authority for one observed view.

    ``mask_bytes`` contains one byte per raster sample (0/1). ``mask_sha256`` is
    an integrity hash of those exact bytes. ``source_alpha_sha256`` can carry the
    upstream image/observation hash without conflating it with this packed mask.
    """

    view_index: int
    width: int
    height: int
    mask_bytes: bytes
    mask_sha256: str
    source_alpha_sha256: str = ""

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Sequence[object]],
        *,
        view_index: int,
        source_alpha_sha256: str = "",
    ) -> "ObservationRasterDomain":
        if not (0 <= int(view_index) < 8):
            raise ValueError("view_index must be in [0,7]")
        width, height, packed = _mask_bytes(rows)
        digest = hashlib.sha256(packed).hexdigest()
        return cls(int(view_index), width, height, packed, digest, str(source_alpha_sha256))

    def validate(self) -> None:
        if not (0 <= int(self.view_index) < 8):
            raise ValueError("view_index must be in [0,7]")
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("observation domain dimensions must be positive")
        expected = int(self.width) * int(self.height)
        if len(self.mask_bytes) != expected:
            raise ValueError(f"observation domain byte count mismatch:{len(self.mask_bytes)}!={expected}")
        if any(v not in (0, 1) for v in self.mask_bytes):
            raise ValueError("observation domain bytes must be binary 0/1")
        digest = hashlib.sha256(self.mask_bytes).hexdigest()
        if digest != self.mask_sha256:
            raise ValueError(f"observation domain hash mismatch:{digest}")

    @property
    def foreground_count(self) -> int:
        return int(sum(self.mask_bytes))

    def contains_pixel(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        return bool(self.mask_bytes[int(y) * int(self.width) + int(x)])

    def triangle_pixels(self, tri: Triangle2) -> set[tuple[int, int]]:
        indices = _triangle_indices(tri, width=self.width, height=self.height)
        if indices is None:
            return _scalar_triangle_pixels(tri, width=self.width, height=self.height)
        rr, cc = indices
        return set(zip(cc.tolist(), rr.tolist()))

    def triangle_inside(self, tri: Triangle2) -> bool:
        indices = _triangle_indices(tri, width=self.width, height=self.height)
        if indices is None:
            pixels = _scalar_triangle_pixels(tri, width=self.width, height=self.height)
            return bool(pixels) and all(self.contains_pixel(x, y) for x, y in pixels)
        rr, cc = indices
        if len(rr) == 0:
            return False
        authority = np.frombuffer(self.mask_bytes, dtype=np.uint8).reshape(self.height, self.width)
        return bool(np.all(authority[rr, cc] != 0))

    def coverage(self, triangles: Iterable[Triangle2]) -> dict:
        if _raster_polygon is None:
            predicted_set: set[tuple[int, int]] = set()
            for tri in triangles:
                predicted_set.update(_scalar_triangle_pixels(tri, width=self.width, height=self.height))
            predicted_bytes = bytearray(self.width * self.height)
            for x, y in predicted_set:
                predicted_bytes[y * self.width + x] = 1
            inside = sum(1 for x, y in predicted_set if self.contains_pixel(x, y))
            foreground = self.foreground_count
            predicted_count = len(predicted_set)
        else:
            predicted = np.zeros((self.height, self.width), dtype=np.bool_)
            for tri in triangles:
                rr, cc = _triangle_indices(tri, width=self.width, height=self.height)
                predicted[rr, cc] = True
            authority = np.frombuffer(self.mask_bytes, dtype=np.uint8).reshape(self.height, self.width) != 0
            inside = int(np.count_nonzero(predicted & authority))
            foreground = int(np.count_nonzero(authority))
            predicted_count = int(np.count_nonzero(predicted))
            predicted_bytes = bytearray(predicted.astype(np.uint8).tobytes())

        precision = 1.0 if predicted_count == 0 else float(inside) / float(predicted_count)
        recall = 1.0 if foreground == 0 else float(inside) / float(foreground)
        union = foreground + predicted_count - inside
        iou = 1.0 if union == 0 else float(inside) / float(union)
        f1 = 1.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        component_metrics = _connected_component_metrics(
            self.mask_bytes,
            bytes(predicted_bytes),
            width=self.width,
            height=self.height,
        )
        return {
            "predicted_pixel_count": int(predicted_count),
            "inside_alpha_pixel_count": int(inside),
            "foreground_pixel_count": int(foreground),
            "precision_inside_alpha": float(precision),
            "source_alpha_recall": float(recall),
            "alpha_iou": float(iou),
            "alpha_f1": float(f1),
            **component_metrics,
        }
