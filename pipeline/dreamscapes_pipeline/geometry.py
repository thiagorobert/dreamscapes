"""Polygon geometry in normalised image coordinates (x, y in 0-1, origin top-left).

Because both axes are normalised by the image size, an area computed here is directly the
fraction of the frame that the polygon covers.
"""

from __future__ import annotations

from collections.abc import Sequence

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


def signed_area(polygon: Sequence[Sequence[float]]) -> float:
    """Shoelace formula. Positive for counter-clockwise in a y-up frame."""
    total = 0.0
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def area(polygon: Sequence[Sequence[float]]) -> float:
    return abs(signed_area(polygon))


def centroid(polygon: Sequence[Sequence[float]]) -> Point:
    """Area-weighted centroid; falls back to the vertex mean for degenerate polygons."""
    a = signed_area(polygon)
    n = len(polygon)
    if n == 0:
        raise ValueError("centroid of an empty polygon is undefined")
    if abs(a) < 1e-12:
        return (
            sum(p[0] for p in polygon) / n,
            sum(p[1] for p in polygon) / n,
        )
    cx = cy = 0.0
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    return (cx / (6.0 * a), cy / (6.0 * a))


def bbox(polygon: Sequence[Sequence[float]]) -> BBox:
    """Axis-aligned bounding box as (x0, y0, x1, y1)."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def _orientation(p: Sequence[float], q: Sequence[float], r: Sequence[float]) -> int:
    value = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(value) < 1e-15:
        return 0
    return 1 if value > 0 else 2


def _on_segment(p: Sequence[float], q: Sequence[float], r: Sequence[float]) -> bool:
    """True when q lies on segment pr, given that p, q, r are collinear."""
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])


def segments_intersect(
    p1: Sequence[float], q1: Sequence[float], p2: Sequence[float], q2: Sequence[float]
) -> bool:
    """True when closed segments p1q1 and p2q2 share at least one point."""
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    return o4 == 0 and _on_segment(p2, q1, q2)


def simplicity_problem(polygon: Sequence[Sequence[float]]) -> str | None:
    """Describe why the polygon is not simple, or return None when it is.

    A simple polygon has no repeated consecutive vertices (including a closing vertex equal to
    the first one), no edges that cross or touch except adjacent edges at their shared vertex,
    and adjacent edges that do not fold back over each other.
    """
    n = len(polygon)
    if n < 3:
        return f"needs at least 3 vertices, has {n}"
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        if a[0] == b[0] and a[1] == b[1]:
            return f"vertices {i} and {(i + 1) % n} are identical"
    edges = [(polygon[i], polygon[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            adjacent = j == i + 1 or (i == 0 and j == n - 1)
            (a1, b1), (a2, b2) = edges[i], edges[j]
            if adjacent:
                # Adjacent edges share exactly one vertex; they are invalid only when collinear
                # and overlapping (the polygon doubles back on itself).
                shared, other_i, other_j = (b1, a1, b2) if j == i + 1 else (a1, b1, a2)
                if _orientation(other_i, shared, other_j) == 0:
                    di = (other_i[0] - shared[0], other_i[1] - shared[1])
                    dj = (other_j[0] - shared[0], other_j[1] - shared[1])
                    if di[0] * dj[0] + di[1] * dj[1] > 0:
                        return f"edges {i} and {j} fold back over each other"
                continue
            if segments_intersect(a1, b1, a2, b2):
                return f"edges {i} and {j} intersect"
    return None


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Even-odd ray casting test."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < x_cross:
                inside = not inside
    return inside
