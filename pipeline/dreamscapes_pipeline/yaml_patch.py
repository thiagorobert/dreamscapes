"""Change a few fields of one scene in ``content/scenes.yaml`` without rewriting the file.

Loading the YAML into Python and dumping it again would lose every comment and re-flow text
that people wrote by hand. Instead, this module finds the exact lines that hold the fields being
changed (PyYAML's node tree records where each node starts and ends) and replaces only those
lines. Everything else, including comments, blank lines, key order and quoting, stays byte for
byte as it was.

Each patch is checked by parsing the result: the new document must equal the old one with the
requested changes applied. If the file is laid out in a way the line-level edit cannot handle
(for example a section written in flow style, ``bridge: {object: ...}``), the patch falls back to
rewriting the whole document with the manifest's own YAML style. That keeps key order but drops
comments, and the caller is told so.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import yaml

from .manifest import _ManifestDumper

# Folded text wraps at this column, the same width `dump_manifest` uses for the whole file.
_WIDTH = 96


class PatchError(ValueError):
    """The requested change could not be applied by editing lines in place."""


@dataclass(frozen=True)
class PatchResult:
    text: str
    in_place: bool
    """False when the document had to be re-dumped, which drops comments."""


def _mapping_get(node: yaml.Node, key: str) -> tuple[yaml.Node, yaml.Node] | None:
    if not isinstance(node, yaml.MappingNode):
        return None
    for key_node, value_node in node.value:
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == key:
            return key_node, value_node
    return None


def _find_section(root: yaml.Node, scene_id: str, section: str) -> yaml.MappingNode:
    scenes = _mapping_get(root, "scenes")
    if scenes is None or not isinstance(scenes[1], yaml.SequenceNode):
        raise PatchError("the document has no 'scenes' list")
    for item in scenes[1].value:
        found = _mapping_get(item, "id")
        if found is None or not isinstance(found[1], yaml.ScalarNode):
            continue
        if found[1].value == scene_id:
            pair = _mapping_get(item, section)
            if pair is None:
                raise PatchError(f"scene '{scene_id}' has no '{section}' mapping")
            node = pair[1]
            if not isinstance(node, yaml.MappingNode) or node.flow_style or not node.value:
                raise PatchError(f"'{section}' of scene '{scene_id}' is not a block mapping")
            return node
    raise PatchError(f"scene '{scene_id}' not found")


def _is_filler(line: str) -> bool:
    """Blank and comment-only lines belong to whatever follows them, not to a field."""
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _last_line(node: yaml.Node, lines: Sequence[str]) -> int:
    """Index of the last line that holds part of ``node``.

    Scalars and flow collections know where they end. A node whose end mark sits after nothing
    but indentation actually ended on an earlier line: block scalars (``>-``, ``|``) consume the
    line breaks and indentation that follow them.
    """
    if isinstance(node, yaml.MappingNode) and not node.flow_style and node.value:
        return max(_last_line(n, lines) for pair in node.value for n in pair)
    if isinstance(node, yaml.SequenceNode) and not node.flow_style and node.value:
        return max(_last_line(n, lines) for n in node.value)
    mark = node.end_mark
    line = mark.line
    if line >= len(lines) or not lines[line][: mark.column].strip():
        line -= 1
    while line > node.start_mark.line and _is_filler(lines[line]):
        line -= 1
    return max(line, node.start_mark.line)


def _render(key: str, value: Any, indent: int) -> list[str]:
    text = yaml.dump(
        {key: value},
        Dumper=_ManifestDumper,
        sort_keys=False,
        allow_unicode=True,
        width=max(40, _WIDTH - indent),
    )
    return [(" " * indent + line) if line.strip() else line for line in text.splitlines(True)]


def render_fields(updates: Mapping[str, Any], indent: int) -> str:
    """The lines `set_scene_fields` writes for ``updates`` (fields set to None are skipped)."""
    return "".join(
        line
        for key, value in updates.items()
        if value is not None
        for line in _render(key, value, indent)
    )


def _rank(order: Sequence[str]):
    position = {key: i for i, key in enumerate(order)}
    return lambda key: position.get(key, len(order))


def _insertion_anchors(
    existing: Sequence[str], new: Sequence[str], order: Sequence[str]
) -> dict[str, str | None]:
    """For each new key, the existing key it goes right after (None: before the first one).

    New keys are placed where ``order`` (the model's field order) says they belong relative to
    the keys already present, so a file written by a person and one written by `dump_manifest`
    both end up in the canonical order without any existing line moving.
    """
    rank = _rank(order)
    known = set(order)
    anchors: dict[str, str | None] = {}
    for key in new:
        before = [k for k in existing if k in known and rank(k) < rank(key)]
        if before:
            anchors[key] = max(before, key=rank)
        elif any(rank(k) > rank(key) for k in existing) or not existing:
            anchors[key] = None
        else:
            anchors[key] = existing[-1]
    return anchors


def _final_keys(existing: Sequence[str], updates: Mapping[str, Any], order: Sequence[str]):
    """Key order of the section after the patch."""
    rank = _rank(order)
    new_keys = [k for k, v in updates.items() if k not in existing and v is not None]
    removed = {k for k, v in updates.items() if v is None}
    anchors = _insertion_anchors(existing, new_keys, order)
    keys = sorted((k for k in new_keys if anchors[k] is None), key=rank)
    for key in existing:
        keys.append(key)
        keys.extend(sorted((k for k in new_keys if anchors[k] == key), key=rank))
    return [k for k in keys if k not in removed]


def _apply_to_data(
    data: Any, scene_id: str, section: str, updates: Mapping[str, Any], order: Sequence[str]
) -> Any:
    """The document as it should read after the patch, with keys in their final order."""
    expected = copy.deepcopy(data)
    for scene in expected.get("scenes", []) if isinstance(expected, dict) else []:
        if isinstance(scene, dict) and scene.get("id") == scene_id:
            target = scene.get(section)
            if not isinstance(target, dict):
                raise PatchError(f"scene '{scene_id}' has no '{section}' mapping")
            merged = {**target, **updates}
            scene[section] = {k: merged[k] for k in _final_keys(list(target), updates, order)}
            return expected
    raise PatchError(f"scene '{scene_id}' not found")


def _patch_lines(
    text: str,
    scene_id: str,
    section: str,
    updates: Mapping[str, Any],
    order: Sequence[str],
) -> str:
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as error:
        raise PatchError(f"invalid YAML: {error}") from error
    if root is None:
        raise PatchError("the document is empty")
    node = _find_section(root, scene_id, section)
    lines = text.splitlines(True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    indent = node.value[0][0].start_mark.column

    spans: dict[str, tuple[int, int]] = {}
    for key_node, value_node in node.value:
        if not isinstance(key_node, yaml.ScalarNode):
            raise PatchError(f"'{section}' has a non-string key")
        spans[key_node.value] = (key_node.start_mark.line, _last_line(value_node, lines) + 1)

    replacements: dict[int, tuple[int, list[str]]] = {}
    for key, value in updates.items():
        if key in spans:
            start, end = spans[key]
            replacements[start] = (end, [] if value is None else _render(key, value, indent))

    new_keys = [k for k, v in updates.items() if k not in spans and v is not None]
    existing = list(spans)
    anchors = _insertion_anchors(existing, new_keys, order)
    insertions: dict[int, list[str]] = {}
    for key in sorted(new_keys, key=_rank(order)):
        anchor = anchors[key]
        position = spans[existing[0]][0] if anchor is None else spans[anchor][1]
        insertions.setdefault(position, []).extend(_render(key, updates[key], indent))

    out: list[str] = []
    i = 0
    while i <= len(lines):
        out.extend(insertions.get(i, []))
        if i == len(lines):
            break
        if i in replacements:
            end, new_lines = replacements[i]
            out.extend(new_lines)
            i = end
            continue
        out.append(lines[i])
        i += 1
    return "".join(out)


def set_scene_fields(
    text: str,
    scene_id: str,
    section: str,
    updates: Mapping[str, Any],
    order: Sequence[str],
) -> PatchResult:
    """Set ``scenes[id == scene_id][section][key] = value`` for every item in ``updates``.

    A value of None removes the key. ``order`` is the canonical field order of the section and
    decides where keys that do not exist yet are inserted. Values must be plain YAML data
    (dicts, lists, strings, numbers, booleans).
    """
    original = yaml.safe_load(text)
    expected = _apply_to_data(original, scene_id, section, updates, order)
    try:
        patched = _patch_lines(text, scene_id, section, updates, order)
        if yaml.safe_load(patched) == expected:
            return PatchResult(patched, in_place=True)
    except (PatchError, yaml.YAMLError):
        pass
    # Whole-document fallback: the same key order as the patched document would have had.
    redumped = yaml.dump(
        expected, Dumper=_ManifestDumper, sort_keys=False, allow_unicode=True, width=_WIDTH
    )
    if yaml.safe_load(redumped) != expected:
        raise PatchError("could not write the change without altering other content")
    return PatchResult(redumped, in_place=False)
