from __future__ import annotations

import pytest

from dreamscapes_pipeline import geometry
from tests.conftest import square


def test_area_centroid_and_bbox_of_a_square():
    poly = square(0.5, 0.4, 0.1)
    assert geometry.area(poly) == pytest.approx(0.04)
    assert geometry.centroid(poly) == pytest.approx((0.5, 0.4))
    assert geometry.bbox(poly) == pytest.approx((0.4, 0.3, 0.6, 0.5))


def test_centroid_is_area_weighted_not_the_vertex_mean():
    # Extra vertices bunched on one side must not pull the centroid over.
    poly = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.01), (1.0, 1.0), (0.0, 1.0)]
    assert geometry.centroid(poly) == pytest.approx((0.5, 0.5))


@pytest.mark.parametrize(
    ("polygon", "expected"),
    [
        ([(0.1, 0.1), (0.3, 0.1), (0.2, 0.3)], None),
        ([(0.1, 0.1), (0.3, 0.3), (0.3, 0.1), (0.1, 0.3)], "intersect"),  # bow tie
        ([(0.1, 0.1), (0.3, 0.1), (0.2, 0.3), (0.1, 0.1)], "identical"),  # closing vertex
        ([(0.1, 0.1), (0.3, 0.1), (0.2, 0.1), (0.2, 0.3)], "fold back"),  # spike along an edge
        ([(0.1, 0.1), (0.2, 0.2)], "at least 3"),
    ],
)
def test_simplicity(polygon, expected):
    problem = geometry.simplicity_problem(polygon)
    if expected is None:
        assert problem is None
    else:
        assert problem is not None and expected in problem


def test_concave_star_is_simple_and_contains_its_centre():
    import math

    star = []
    for k in range(10):
        r = 0.1 if k % 2 == 0 else 0.04
        a = -math.pi / 2 + math.pi * k / 5
        star.append((0.5 + r * math.cos(a), 0.5 + r * math.sin(a)))
    assert geometry.simplicity_problem(star) is None
    assert geometry.point_in_polygon(geometry.centroid(star), star)
