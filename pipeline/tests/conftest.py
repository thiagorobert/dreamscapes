from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from dreamscapes_pipeline.layout import MASTER_HEIGHT, MASTER_WIDTH, RepoPaths
from dreamscapes_pipeline.manifest import Bridge, Manifest, Scene, save_manifest
from dreamscapes_pipeline.placeholders import run_placeholders


def square(cx: float, cy: float, half: float = 0.1) -> list[tuple[float, float]]:
    """A square in normalised coordinates; half=0.1 covers 4% of the frame."""
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


def good_manifest() -> Manifest:
    """Three scenes in one loop, each describing the previous scene's bridge object."""
    specs = [
        ("first-scene", "lantern", "second-scene", "A quiet room lit by one candle."),
        ("second-scene", "paper boat", "third-scene", "A pond where a lantern floats."),
        ("third-scene", "candle", "first-scene", "A harbour full of paper boats."),
    ]
    scenes = [
        Scene(
            id=sid,
            title=sid.replace("-", " ").title(),
            alt=f"Alt text for {sid}.",
            scene=description,
            bridge=Bridge(object=obj, placement="small, centre", polygon=square(0.5, 0.5)),
            next=nxt,
        )
        for sid, obj, nxt, description in specs
    ]
    return Manifest(start="first-scene", style_anchor="first-scene", scenes=scenes)


def make_repo_root(root: Path) -> RepoPaths:
    """Create the two markers that identify a Dreamscapes checkout."""
    (root / "content").mkdir(parents=True, exist_ok=True)
    (root / "pipeline").mkdir(parents=True, exist_ok=True)
    (root / "pipeline" / "pyproject.toml").write_text('[project]\nname = "dreamscapes-pipeline"\n')
    return RepoPaths(root)


def write_small_repo(root: Path) -> RepoPaths:
    """A repository with the three-scene manifest and solid-colour masters, no exports."""
    paths = make_repo_root(root)
    manifest = good_manifest()
    save_manifest(manifest, paths.scenes_yaml)
    paths.masters.mkdir(parents=True, exist_ok=True)
    for scene in manifest.scenes:
        Image.new("RGB", (MASTER_WIDTH, MASTER_HEIGHT), (90, 120, 140)).save(paths.master(scene.id))
    return paths


@pytest.fixture
def small_repo(tmp_path: Path) -> RepoPaths:
    return write_small_repo(tmp_path / "repo")


@pytest.fixture(scope="session")
def _placeholder_source(tmp_path_factory: pytest.TempPathFactory) -> RepoPaths:
    paths = make_repo_root(tmp_path_factory.mktemp("placeholders"))
    run_placeholders(paths)
    return paths


@pytest.fixture
def placeholder_repo(_placeholder_source: RepoPaths, tmp_path: Path) -> RepoPaths:
    """A private copy of a repository filled by `dsp placeholders`, safe to modify."""
    root = tmp_path / "repo"
    shutil.copytree(_placeholder_source.root, root)
    return RepoPaths(root)


def file_hashes(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }
