from __future__ import annotations

from pathlib import Path

import pytest

from dreamscapes_pipeline.layout import RepoRootNotFound, find_repo_root
from tests.conftest import make_repo_root


def test_finds_the_root_from_a_nested_directory(tmp_path: Path):
    make_repo_root(tmp_path)
    nested = tmp_path / "web" / "public" / "dreams"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == tmp_path.resolve()


def test_an_unrelated_site_with_a_content_directory_is_not_a_root(tmp_path: Path):
    blog = tmp_path / "blog"  # e.g. a Hugo or Astro project
    (blog / "content").mkdir(parents=True)
    (blog / "posts").mkdir()
    with pytest.raises(RepoRootNotFound, match="pipeline/pyproject.toml"):
        find_repo_root(blog / "posts")


def test_an_unrelated_content_directory_inside_the_repo_is_skipped(tmp_path: Path):
    make_repo_root(tmp_path)
    inner = tmp_path / "web" / "docs-site"
    (inner / "content" / "posts").mkdir(parents=True)
    assert find_repo_root(inner / "content" / "posts") == tmp_path.resolve()


@pytest.mark.parametrize("keep", ["content", "pyproject"])
def test_both_markers_are_required(tmp_path: Path, keep: str):
    if keep == "content":
        (tmp_path / "content").mkdir()
    else:
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "pyproject.toml").write_text("")
    with pytest.raises(RepoRootNotFound):
        find_repo_root(tmp_path)
