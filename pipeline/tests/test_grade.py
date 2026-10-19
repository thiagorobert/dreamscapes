from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image

from dreamscapes_pipeline.grade import (
    GradeConfigError,
    GradeParams,
    binomial_kernel,
    gaussian_blur,
    grade_array,
    load_grade_params,
    parse_grade_params,
    run_grade,
    scene_seed,
)
from dreamscapes_pipeline.layout import MASTER_HEIGHT, MASTER_WIDTH
from dreamscapes_pipeline.manifest import load_manifest
from dreamscapes_pipeline.placeholders import is_placeholder_image
from tests.conftest import file_hashes
from tests.fakes import offline, write_master  # noqa: F401

# content/grade.toml as the content package wrote it; the grade must read exactly these keys.
GRADE_TOML = """
[desaturate]
saturation = 0.85

[tone]
black_lift = 18
highlight_shoulder = 0.8

[split_tone]
shadow_colour = [64, 128, 132]
highlight_colour = [228, 170, 96]
strength = 0.12

[bloom]
threshold = 0.8
radius = 0.02
mix = 0.25

[vignette]
strength = 0.15

[grain]
sigma = 2.5
"""

GOLDEN_INPUT_SHA256 = "de8c5265e935fb9f130080f8fde0aa744f06f067cd0bdc950aa2c50a6a1ed9ce"
GOLDEN_OUTPUT_SHA256 = "1f054a906f35cf8de2ced5c195b0b4bee67fd8328600579ff0fdfdf7ef9505be"


def golden_input() -> np.ndarray:
    """A fixed 512x288 test card: two ramps, a texture and one bright patch to bloom."""
    h, w = 288, 512
    y, x = np.mgrid[0:h, 0:w]
    image = np.stack([(x * 255) // (w - 1), (y * 255) // (h - 1), ((x + y) * 7) % 256], -1)
    image = image.astype(np.uint8)
    image[100:140, 300:360] = 250
    return image


def test_golden_output():
    """Any change to the grade's maths changes this hash; update it only on purpose."""
    image = golden_input()
    assert hashlib.sha256(image.tobytes()).hexdigest() == GOLDEN_INPUT_SHA256
    graded = grade_array(image, GradeParams(), scene_seed("golden-scene"))
    assert graded.shape == image.shape and graded.dtype == np.uint8
    assert hashlib.sha256(graded.tobytes()).hexdigest() == GOLDEN_OUTPUT_SHA256


def test_grading_is_repeatable_and_grain_depends_on_the_scene():
    image = golden_input()
    a = grade_array(image, GradeParams(), scene_seed("a"))
    assert np.array_equal(a, grade_array(image, GradeParams(), scene_seed("a")))
    b = grade_array(image, GradeParams(), scene_seed("b"))
    assert not np.array_equal(a, b)
    no_grain = GradeParams.model_validate({"grain": {"sigma": 0}})
    assert np.array_equal(
        grade_array(image, no_grain, scene_seed("a")), grade_array(image, no_grain, scene_seed("b"))
    )


def test_neutral_parameters_leave_the_image_alone():
    neutral = GradeParams.model_validate(
        {
            "desaturate": {"saturation": 1.0},
            "tone": {"black_lift": 0, "highlight_shoulder": 1.0},
            "split_tone": {"strength": 0},
            "bloom": {"mix": 0},
            "vignette": {"strength": 0},
            "grain": {"sigma": 0},
        }
    )
    image = golden_input()
    assert np.array_equal(grade_array(image, neutral, 1), image)


def test_each_step_moves_the_image_the_right_way():
    flat = np.full((90, 160, 3), (200, 60, 40), dtype=np.uint8)
    graded = grade_array(flat, GradeParams.model_validate({"grain": {"sigma": 0}}), 1)
    centre, corner = graded[45, 80].astype(int), graded[0, 0].astype(int)
    assert np.ptp(centre) < np.ptp(flat[45, 80].astype(int)), "desaturated"
    assert corner.sum() < centre.sum(), "vignette darkens the corners"
    black = grade_array(
        np.zeros((90, 160, 3), np.uint8),
        GradeParams.model_validate(
            {"vignette": {"strength": 0}, "grain": {"sigma": 0}, "split_tone": {"strength": 0}}
        ),
        1,
    )
    assert set(np.unique(black)) == {18}, "black is lifted to black_lift"
    white = grade_array(
        np.full((90, 160, 3), 255, np.uint8),
        GradeParams.model_validate(
            {
                "vignette": {"strength": 0},
                "grain": {"sigma": 0},
                "split_tone": {"strength": 0},
                "bloom": {"mix": 0},
            }
        ),
        1,
    )
    assert 235 < int(white[0, 0, 0]) < 250, "highlights roll off below pure white"


def test_blur_kernel_and_blur():
    kernel = binomial_kernel(5.0)
    assert kernel.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(kernel, kernel[::-1]) and kernel.argmax() == len(kernel) // 2
    variance = float(((np.arange(len(kernel)) - len(kernel) // 2) ** 2 * kernel).sum())
    assert variance == pytest.approx(25.0, rel=0.05)
    constant = np.full((64, 96, 3), 0.5)
    for sigma in (0.0, 3.0, 40.0):
        assert np.allclose(gaussian_blur(constant, sigma), 0.5)
    spot = np.zeros((128, 128, 1))
    spot[64, 64] = 1.0
    blurred = gaussian_blur(spot, 20.0)
    assert blurred.shape == spot.shape and blurred[64, 64] < 0.01 and blurred.sum() > 0.5


def test_parameters_from_the_content_file(small_repo):
    small_repo.grade_toml.write_text(GRADE_TOML)
    loaded = load_grade_params(small_repo)
    assert loaded.source == "content/grade.toml" and loaded.warnings == []
    assert loaded.params == GradeParams(), "the shipped defaults equal content/grade.toml"
    small_repo.grade_toml.write_text("[grain]\nsigma = 4\nseed = 3\n\n[sparkle]\non = true\n")
    loaded = load_grade_params(small_repo)
    assert loaded.params.grain.sigma == 4 and loaded.params.tone.black_lift == 18
    assert sorted(loaded.warnings) == [
        "content/grade.toml: unknown key 'grain.seed' is ignored by the grade",
        "content/grade.toml: unknown key 'sparkle' is ignored by the grade",
    ]


def test_missing_and_invalid_parameter_files(small_repo):
    assert "defaults" in load_grade_params(small_repo).source
    small_repo.grade_toml.write_text("[tone]\nblack_lift = 400\n")
    with pytest.raises(GradeConfigError, match="tone.black_lift"):
        load_grade_params(small_repo)
    small_repo.grade_toml.write_text("[tone\n")
    with pytest.raises(GradeConfigError, match="not valid TOML"):
        load_grade_params(small_repo)
    with pytest.raises(GradeConfigError, match="split_tone.shadow_colour"):
        parse_grade_params({"split_tone": {"shadow_colour": [1, 2]}})


def test_run_grade_writes_real_masters_and_is_idempotent(small_repo):
    manifest = load_manifest(small_repo.scenes_yaml)
    write_master(small_repo, "second-scene", (180, 140, 90))
    small_repo.master("third-scene").unlink()
    report = run_grade(small_repo, manifest)
    assert report.written == ["first-scene", "second-scene"]
    assert report.missing == ["third-scene"]
    with Image.open(small_repo.graded("second-scene")) as graded:
        assert graded.size == (MASTER_WIDTH, MASTER_HEIGHT)
    again = run_grade(small_repo, manifest)
    assert again.written == [] and again.unchanged == ["first-scene", "second-scene"]


def test_placeholder_graded_copies_are_left_alone_and_preview_shows_them(placeholder_repo):
    before = file_hashes(placeholder_repo.root)
    manifest = load_manifest(placeholder_repo.scenes_yaml)
    report = run_grade(placeholder_repo, manifest)
    assert report.written == [] and len(report.placeholders) == 10
    assert file_hashes(placeholder_repo.root) == before

    report = run_grade(placeholder_repo, manifest, preview=True)
    assert report.preview_page == placeholder_repo.review / "grade.html"
    html = report.preview_page.read_text()
    assert html.count("<figcaption") == 20
    assert "built-in defaults" in html
    for _, before_rel, after_rel in report.previews:
        assert (placeholder_repo.review / before_rel).exists()
        assert (placeholder_repo.review / after_rel).exists()
    unchanged = {k: v for k, v in file_hashes(placeholder_repo.root).items() if "review" not in k}
    assert unchanged == before
    assert is_placeholder_image(placeholder_repo.graded("cliff-cottage"))
