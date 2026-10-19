"""Repository layout and image geometry shared by every pipeline command.

All paths are resolved relative to the repository root: the nearest directory at or above the
current working directory that contains both a ``content/`` directory and
``pipeline/pyproject.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Masters are 16:9 landscape PNGs at this size; exports are scaled copies of the graded master.
MASTER_WIDTH = 2048
MASTER_HEIGHT = 1152
EXPORT_WIDTHS: tuple[int, ...] = (2048, 1536, 1024, 640)
EXPORT_FORMATS: tuple[str, ...] = ("avif", "webp")


def export_height(width: int) -> int:
    """Height of an export of the given width, keeping the master's aspect ratio."""
    return round(width * MASTER_HEIGHT / MASTER_WIDTH)


class RepoRootNotFound(RuntimeError):
    """Raised when no Dreamscapes repository root exists at or above the start path."""


def is_repo_root(path: Path) -> bool:
    """True for a Dreamscapes checkout.

    A ``content/`` directory alone is not enough: static-site generators such as Hugo or Astro
    use the same name, and treating such a project as the root would make `dsp` write into it
    and load its ``.env``. Requiring the pipeline's own ``pyproject.toml`` as well rules that out.
    """
    return (path / "content").is_dir() and (path / "pipeline" / "pyproject.toml").is_file()


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: the current directory) to the repository root."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if is_repo_root(candidate):
            return candidate
    raise RepoRootNotFound(
        "could not find the Dreamscapes repository root: no directory containing both "
        f"'content/' and 'pipeline/pyproject.toml' at or above {here}"
    )


@dataclass(frozen=True)
class RepoPaths:
    """Every path the pipeline reads or writes, derived from the repository root."""

    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> RepoPaths:
        return cls(find_repo_root(start))

    # Authored content
    @property
    def content(self) -> Path:
        return self.root / "content"

    @property
    def scenes_yaml(self) -> Path:
        return self.content / "scenes.yaml"

    @property
    def style_md(self) -> Path:
        return self.content / "style.md"

    @property
    def grade_toml(self) -> Path:
        return self.content / "grade.toml"

    # Pipeline assets
    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def masters(self) -> Path:
        return self.assets / "masters"

    @property
    def masks(self) -> Path:
        return self.assets / "masks"

    @property
    def candidates(self) -> Path:
        return self.assets / "candidates"

    @property
    def review(self) -> Path:
        return self.assets / "review"

    def master(self, scene_id: str) -> Path:
        return self.masters / f"{scene_id}.png"

    def graded(self, scene_id: str) -> Path:
        return self.masters / f"{scene_id}.graded.png"

    def mask(self, scene_id: str) -> Path:
        return self.masks / f"{scene_id}.png"

    def candidate_dir(self, scene_id: str) -> Path:
        return self.candidates / scene_id

    def review_dir(self, scene_id: str) -> Path:
        return self.review / scene_id

    # Pipeline bookkeeping
    @property
    def generation_log(self) -> Path:
        return self.root / "pipeline" / "logs" / "generations.jsonl"

    # Website
    @property
    def web_public(self) -> Path:
        return self.root / "web" / "public"

    @property
    def scenes_json(self) -> Path:
        return self.web_public / "scenes.json"

    @property
    def dreams(self) -> Path:
        return self.web_public / "dreams"

    @property
    def morphs(self) -> Path:
        return self.web_public / "morphs"

    def export_dir(self, scene_id: str) -> Path:
        return self.dreams / scene_id

    def public_file(self, src: str) -> Path:
        """Resolve a ``src`` from the built manifest (relative to the site root) to a file."""
        return self.web_public / src

    # Schema
    @property
    def schema_json(self) -> Path:
        return self.root / "schema" / "scenes.schema.json"
