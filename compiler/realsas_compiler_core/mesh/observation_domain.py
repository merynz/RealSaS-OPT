from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence

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
        xs = [float(p[0]) for p in tri]
        ys = [float(p[1]) for p in tri]
        xmin = max(0, int(math.ceil(min(xs) - 1.0e-9)))
        xmax = min(self.width - 1, int(math.floor(max(xs) + 1.0e-9)))
        ymin = max(0, int(math.ceil(min(ys) - 1.0e-9)))
        ymax = min(self.height - 1, int(math.floor(max(ys) + 1.0e-9)))
        if xmin > xmax or ymin > ymax:
            return set()
        out: set[tuple[int, int]] = set()
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                if _point_in_triangle((float(x), float(y)), tri):
                    out.add((x, y))
        return out

    def triangle_inside(self, tri: Triangle2) -> bool:
        pixels = self.triangle_pixels(tri)
        return bool(pixels) and all(self.contains_pixel(x, y) for x, y in pixels)

    def coverage(self, triangles: Iterable[Triangle2]) -> dict[str, float | int]:
        predicted: set[tuple[int, int]] = set()
        for tri in triangles:
            predicted.update(self.triangle_pixels(tri))
        inside = sum(1 for x, y in predicted if self.contains_pixel(x, y))
        foreground = self.foreground_count
        predicted_count = len(predicted)
        precision = 1.0 if predicted_count == 0 else float(inside) / float(predicted_count)
        recall = 1.0 if foreground == 0 else float(inside) / float(foreground)
        return {
            "predicted_pixel_count": int(predicted_count),
            "inside_alpha_pixel_count": int(inside),
            "foreground_pixel_count": int(foreground),
            "precision_inside_alpha": float(precision),
            "source_alpha_recall": float(recall),
        }
