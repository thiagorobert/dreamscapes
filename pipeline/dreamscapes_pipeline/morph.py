"""`dsp morph`: pre-rendered Veo clips that morph one scene into the next (plan section 7.2).

For an edge A -> B (B is A's ``next``), Veo 3.1 gets the graded master of A as the first frame,
the graded master of B as the last frame, and a prompt made of the style block from
``content/style.md`` plus a short motion paragraph: drift toward A's bridge object, soften into
mist and light, become B. The Veo request shape, model ids, durations, polling and prices were
checked against the live Gemini API docs on 2026-10-19; the details are in
`dreamscapes_pipeline.providers.veo`.

After the download, ffmpeg turns the clip into what the site plays:

* the audio track Veo always generates is dropped (``-an``; the site plays clips muted);
* the video is re-encoded as H.264 (``libx264``, ``yuv420p``, CRF 23, ``+faststart`` so playback
  can begin before the whole file arrives), with metadata stripped;
* a clip over 5 MB at its native 1080p is scaled to 720p rather than starved of bits, and only
  then is the CRF raised a little.

The result is written to ``web/public/morphs/<a>__<b>.<hash>.mp4``, where ``hash`` is the first
eight hex digits of the SHA-256 of the file (the same naming as the image exports, so a browser
never plays a stale clip from its cache). Scene A's authored transition in
``content/scenes.yaml`` becomes ``{type: video, video: morphs/<file>, duration: <seconds>}``,
with the duration measured by ffprobe, and ``web/public/scenes.json`` is re-exported through
`dsp export`'s own code.

Every clip is also reviewed: its first, middle and last frames go to
``assets/review/morphs/<a>__<b>/`` with a difference score against the masters (mean absolute
error, 0-255, on a 256-pixel-wide downscale), and ``assets/review/morphs/index.html`` shows the
frames side by side with the masters. A person (or the orchestrator) still looks at every clip:
Veo can invent motion, people or text that breaks the mood, and a rejected edge simply keeps
``type: zoom``.

Bookkeeping follows the rest of the pipeline: edges whose clip is already in place are skipped
unless ``--force``; every paid call is one ``kind: morph`` row in
``pipeline/logs/generations.jsonl``; the spend guard prints the estimate first and enforces
``DSP_MAX_SPEND_USD``; rate limits and server errors are retried with backoff. The raw Veo
download (with its audio) is kept, git-ignored, in ``assets/candidates/morphs/``, so a clip whose
post-processing failed is finished on the next run without paying again, and ``--reencode``
rebuilds clips from those downloads after a change to the encoding settings.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from . import genlog
from .export import content_hash
from .generate import StyleBlock, StyleBlockError, load_style_block, with_definite_article
from .imaging import atomic_write, sha256_bytes
from .layout import RepoPaths
from .manifest import Manifest, ManifestError, Scene, parse_manifest
from .placeholders import is_placeholder_image
from .providers.base import MissingCredentials, ProviderError, ReferenceImage
from .providers.veo import VeoProvider
from .retry import with_retries
from .spend import SpendGuard, SpendLimitExceeded
from .yaml_patch import PatchError, set_scene_fields

Echo = Callable[[str], None]

DEFAULT_RESOLUTION = "1080p"
DEFAULT_DURATION = 8

MAX_CLIP_BYTES = 5_000_000
"""Upper bound for one committed clip (5 MB)."""
DEFAULT_CRF = 23
ENCODE_LADDER: tuple[tuple[int | None, int], ...] = (
    (None, DEFAULT_CRF),
    (720, DEFAULT_CRF),
    (720, 26),
    (720, 28),
)
"""(target height or None for the source size, CRF), tried in order until a clip fits."""
X264_PRESET = "slow"

POLL_INTERVAL_S = 10.0
"""The docs poll every 10 seconds."""
OPERATION_TIMEOUT_S = 15 * 60.0
"""The docs quote up to 6 minutes at peak; this leaves plenty of room before giving up."""
PROGRESS_EVERY_POLLS = 6

METRIC_SIZE = (256, 144)
STRIP_FRAMES = 8
FRAME_MATCH_WARN = 20.0
"""A first or last frame further than this from its master (0-255 scale) gets a warning."""

TRANSITION_FIELD_ORDER = ("type", "video", "duration")

MOTION_TEMPLATE = (
    "A slow dreamlike dissolve. The camera drifts gently toward {bridge}; the scene softens into "
    "mist and light and becomes {destination}. No text, no people, calm, continuous motion."
)


# --------------------------------------------------------------------------------------------
# Names and paths
# --------------------------------------------------------------------------------------------


def edge_stem(a: str, b: str) -> str:
    return f"{a}__{b}"


def clip_name(a: str, b: str, digest: str) -> str:
    return f"{edge_stem(a, b)}.{digest}.mp4"


def _clip_pattern(a: str, b: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(edge_stem(a, b))}\.[0-9a-f]{{8}}\.mp4$")


def edge_clips(paths: RepoPaths, a: str, b: str) -> list[Path]:
    """Every hashed clip of this edge in ``web/public/morphs``."""
    if not paths.morphs.is_dir():
        return []
    pattern = _clip_pattern(a, b)
    return sorted(p for p in paths.morphs.iterdir() if p.is_file() and pattern.match(p.name))


def raw_dir(paths: RepoPaths) -> Path:
    return paths.candidates / "morphs"


def raw_path(paths: RepoPaths, a: str, b: str) -> Path:
    """The Veo download as received, audio included (git-ignored)."""
    return raw_dir(paths) / f"{edge_stem(a, b)}.veo.mp4"


def review_root(paths: RepoPaths) -> Path:
    return paths.review / "morphs"


def review_dir(paths: RepoPaths, a: str, b: str) -> Path:
    return review_root(paths) / edge_stem(a, b)


def _rel(paths: RepoPaths, path: Path) -> str:
    return path.relative_to(paths.root).as_posix() if path.is_relative_to(paths.root) else str(path)


# --------------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------------


def short_description(scene: Scene) -> str:
    """The scene's alt text as a phrase that can follow "becomes": first letter lower-cased
    (unless it starts an acronym) and the closing full stop dropped."""
    text = " ".join(scene.alt.split()).rstrip(" .")
    if len(text) > 1 and text[0].isupper() and not text[1].isupper():
        text = text[0].lower() + text[1:]
    return text


def build_morph_prompt(style_block: str, origin: Scene, destination: Scene) -> str:
    motion = MOTION_TEMPLATE.format(
        bridge=with_definite_article(origin.bridge.object),
        destination=short_description(destination),
    )
    return f"{style_block.strip()}\n\n{motion}"


# --------------------------------------------------------------------------------------------
# ffmpeg
# --------------------------------------------------------------------------------------------

Runner = Callable[[list[str]], str]
"""Runs one ffmpeg or ffprobe command line and returns its standard output.

Raises `ToolError` when the command fails. Tests substitute a fake."""


class ToolError(RuntimeError):
    """ffmpeg or ffprobe is missing or failed."""


def require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise ToolError(
            f"{' and '.join(missing)} not found on PATH; `dsp morph` needs ffmpeg to strip the "
            "audio and re-encode the clips. Install it with `brew install ffmpeg`."
        )


def subprocess_runner(args: list[str]) -> str:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        raise ToolError(f"{args[0]} not found; install it with `brew install ffmpeg`") from error
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.strip().splitlines()[-8:])
        raise ToolError(f"{args[0]} exited with {completed.returncode}: {tail}")
    return completed.stdout


FFMPEG = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]


@dataclass(frozen=True)
class ClipInfo:
    duration: float
    width: int
    height: int
    codec: str
    pix_fmt: str
    frames: int | None
    audio_streams: int
    size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "pix_fmt": self.pix_fmt,
            "frames": self.frames,
            "audio_streams": self.audio_streams,
            "bytes": self.size,
        }


def probe(runner: Runner, path: Path) -> ClipInfo:
    """Duration, size and streams of a video file, from ffprobe."""
    out = runner(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,pix_fmt,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        data = json.loads(out)
    except json.JSONDecodeError as error:
        raise ToolError(f"ffprobe printed something that is not JSON for {path}") from error
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ToolError(f"{path} has no video stream")
    try:
        duration = round(float((data.get("format") or {})["duration"]), 3)
    except (KeyError, TypeError, ValueError) as error:
        raise ToolError(f"ffprobe reported no duration for {path}") from error
    frames = video.get("nb_frames")
    return ClipInfo(
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        codec=str(video.get("codec_name") or ""),
        pix_fmt=str(video.get("pix_fmt") or ""),
        frames=int(frames) if str(frames or "").isdigit() else None,
        audio_streams=sum(1 for s in streams if s.get("codec_type") == "audio"),
        size=path.stat().st_size,
    )


def encode_command(src: Path, dst: Path, height: int | None, crf: int) -> list[str]:
    """Video only, H.264 in yuv420p, index at the front, no metadata."""
    args = [*FFMPEG, "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1"]
    if height is not None:
        args += ["-vf", f"scale=-2:{height}:flags=lanczos"]
    args += [
        "-c:v",
        "libx264",
        "-preset",
        X264_PRESET,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    return args


@dataclass(frozen=True)
class EncodedClip:
    path: Path
    info: ClipInfo
    crf: int
    scaled_to: int | None
    sha256: str


def encode_clip(
    paths: RepoPaths, runner: Runner, raw: Path, a: str, b: str, echo: Echo = print
) -> EncodedClip:
    """Encode ``raw`` for the web as ``web/public/morphs/<a>__<b>.<hash>.mp4``, no larger than
    `MAX_CLIP_BYTES`.

    Earlier clips of the same edge are removed once the new one is in place.
    """
    source = probe(runner, raw)
    paths.morphs.mkdir(parents=True, exist_ok=True)
    temporary = paths.morphs / f".{edge_stem(a, b)}.encoding.mp4"
    tried: list[str] = []
    seen: set[tuple[int | None, int]] = set()
    try:
        for height, crf in ENCODE_LADDER:
            target = None if height is None or height >= source.height else height
            if (target, crf) in seen:
                continue
            seen.add((target, crf))
            runner(encode_command(raw, temporary, target, crf))
            size = temporary.stat().st_size
            label = f"{target or source.height}p CRF {crf}"
            tried.append(f"{label}: {size / 1e6:.2f} MB")
            if size <= MAX_CLIP_BYTES:
                info = probe(runner, temporary)
                if info.audio_streams:
                    raise ToolError(
                        f"the encoded clip still has {info.audio_streams} audio stream(s)"
                    )
                data = temporary.read_bytes()
                final = paths.morphs / clip_name(a, b, content_hash(data))
                os.replace(temporary, final)
                for stale in edge_clips(paths, a, b):
                    if stale != final:
                        stale.unlink()
                if len(tried) > 1:
                    echo(f"  encoded after {len(tried)} tries ({'; '.join(tried)})")
                return EncodedClip(
                    path=final,
                    info=info,
                    crf=crf,
                    scaled_to=target,
                    sha256=sha256_bytes(data),
                )
            echo(f"  {label} is {size / 1e6:.2f} MB, over {MAX_CLIP_BYTES / 1e6:.0f} MB")
    finally:
        temporary.unlink(missing_ok=True)
    raise ToolError(
        f"could not get {edge_stem(a, b)} under {MAX_CLIP_BYTES / 1e6:.0f} MB ({'; '.join(tried)})"
    )


# --------------------------------------------------------------------------------------------
# Review: frames and difference scores
# --------------------------------------------------------------------------------------------


def extract_frames(runner: Runner, clip: Path, out_dir: Path, duration: float) -> dict[str, Path]:
    """First, middle and last frame of ``clip`` as PNGs in ``out_dir``, plus ``strip.jpg``, a
    4x2 contact sheet of the whole clip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {name: out_dir / f"{name}.png" for name in ("first", "middle", "last")}
    frames["strip"] = out_dir / "strip.jpg"
    # Frames of an earlier take must never pass for this one's if ffmpeg writes nothing.
    for path in frames.values():
        path.unlink(missing_ok=True)
    runner([*FFMPEG, "-i", str(clip), "-frames:v", "1", str(frames["first"])])
    runner(
        [*FFMPEG, "-ss", f"{duration / 2:.3f}", "-i", str(clip), "-frames:v", "1"]
        + [str(frames["middle"])]
    )
    # Seeking to half a second before the end and letting every later frame overwrite the same
    # file leaves exactly the last frame, without having to know the frame count.
    runner([*FFMPEG, "-sseof", "-0.5", "-i", str(clip), "-update", "1", str(frames["last"])])
    # One small frame per eighth of the clip, tiled 4x2: the quickest way for a reviewer to see
    # everything Veo put between the two stills (people or text appearing mid-way, say).
    rate = STRIP_FRAMES / max(duration, 0.001)
    runner(
        [*FFMPEG, "-i", str(clip), "-vf", f"fps={rate:.6f},scale=480:-2,tile=4x2"]
        + ["-frames:v", "1", "-q:v", "3", str(frames["strip"])]
    )
    for name, path in frames.items():
        if not path.exists():
            raise ToolError(f"ffmpeg did not write the {name} frame of {clip}")
    return frames


def _small(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("RGB").resize(METRIC_SIZE, Image.Resampling.BOX)
    return np.asarray(image, dtype=np.float64)


def frame_difference(a: Path, b: Path) -> float:
    """Mean absolute difference (0-255) between two images, both shrunk to 256x144 first so
    film grain, compression noise and the clip's lower resolution do not count."""
    return round(float(np.abs(_small(a) - _small(b)).mean()), 2)


@dataclass
class EdgeReview:
    a: str
    b: str
    clip: str
    duration: float
    width: int
    height: int
    bytes: int
    crf: int
    first_vs_a: float
    last_vs_b: float
    first_vs_b: float
    last_vs_a: float
    model: str | None = None
    cost_usd: float | None = None
    log_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def warnings(self) -> list[str]:
        out = []
        if self.first_vs_a > FRAME_MATCH_WARN:
            out.append(f"first frame is {self.first_vs_a} from master {self.a}")
        if self.last_vs_b > FRAME_MATCH_WARN:
            out.append(f"last frame is {self.last_vs_b} from master {self.b}")
        return out


def review_clip(
    paths: RepoPaths,
    runner: Runner,
    a: str,
    b: str,
    encoded: EncodedClip,
    *,
    model: str | None,
    cost: float | None,
    log_id: str | None,
) -> EdgeReview:
    frames = extract_frames(runner, encoded.path, review_dir(paths, a, b), encoded.info.duration)
    master_a, master_b = paths.graded(a), paths.graded(b)
    review = EdgeReview(
        a=a,
        b=b,
        clip=encoded.path.relative_to(paths.web_public).as_posix(),
        duration=encoded.info.duration,
        width=encoded.info.width,
        height=encoded.info.height,
        bytes=encoded.info.size,
        crf=encoded.crf,
        first_vs_a=frame_difference(frames["first"], master_a),
        last_vs_b=frame_difference(frames["last"], master_b),
        first_vs_b=frame_difference(frames["first"], master_b),
        last_vs_a=frame_difference(frames["last"], master_a),
        model=model,
        cost_usd=cost,
        log_id=log_id,
    )
    target = review_dir(paths, a, b) / "review.json"
    target.write_text(json.dumps(review.as_dict(), indent=2) + "\n", encoding="utf-8")
    return review


def write_review_index(paths: RepoPaths, manifest: Manifest) -> Path:
    """``assets/review/morphs/index.html``: every reviewed edge in loop order."""
    from .review import esc, page, url

    here = review_root(paths)
    here.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for scene in loop_order(manifest):
        a, b = scene.id, scene.next
        data_path = review_dir(paths, a, b) / "review.json"
        if not data_path.exists():
            continue
        data = json.loads(data_path.read_text(encoding="utf-8"))
        folder = review_dir(paths, a, b)
        in_manifest = scene.transition.type == "video" and scene.transition.video == data["clip"]
        status = (
            '<span class="badge pass">in the manifest</span>'
            if in_manifest
            else '<span class="badge fail">not in the manifest (zoom)</span>'
        )
        cells = [
            ("master " + a, paths.graded(a)),
            ("first frame", folder / "first.png"),
            ("middle frame", folder / "middle.png"),
            ("last frame", folder / "last.png"),
            ("master " + b, paths.graded(b)),
        ]
        figures = "".join(
            f'<figure><img loading="lazy" src="{url(here, p)}" alt="{esc(label)}">'
            f"<figcaption>{esc(label)}</figcaption></figure>"
            for label, p in cells
        )
        clip = paths.public_file(data["clip"])
        rows.append(
            f"<h2>{esc(a)} &rarr; {esc(b)}</h2>\n"
            f"<p>{status} <code>{esc(data['clip'])}</code> &middot; {data['duration']} s &middot; "
            f"{data['width']}x{data['height']} &middot; CRF {data['crf']} &middot; "
            f"{data['bytes'] / 1e6:.2f} MB</p>\n"
            f"<p>first frame vs master {esc(a)}: <b>{data['first_vs_a']}</b> "
            f"(vs master {esc(b)}: {data['first_vs_b']}) &middot; last frame vs master "
            f"{esc(b)}: <b>{data['last_vs_b']}</b> (vs master {esc(a)}: {data['last_vs_a']})"
            f' <span class="muted">mean absolute difference, 0-255, at 256x144</span></p>\n'
            f'<div class="strip">{figures}</div>\n'
            f'<p class="muted">the whole clip, eight frames evenly spaced:</p>'
            f'<img class="contact" loading="lazy" src="{url(here, folder / "strip.jpg")}" '
            'alt="contact sheet of the clip">\n'
            f'<video src="{url(here, clip)}" controls muted playsinline preload="none"></video>'
        )
    body = (
        "<h1>Morph clips</h1>\n<p class=muted>Each row: the outgoing master, the clip's first, "
        "middle and last frames, and the incoming master. Lower differences mean the clip "
        "starts and ends on the stills; the numbers in brackets compare against the other "
        "master, for scale.</p>\n" + ("\n".join(rows) or "<p>No clips reviewed yet.</p>")
    )
    extra = (
        "<style>.contact, main video { width: 100%; max-width: 960px; display: block; "
        "margin-top: 8px; border-radius: 4px; }</style>\n"
    )
    target = here / "index.html"
    target.write_text(page("Morph clips", extra + body), encoding="utf-8")
    return target


# --------------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------------


def loop_order(manifest: Manifest) -> list[Scene]:
    """Scenes from the start scene along ``next`` pointers, then any not on that path."""
    by_id = manifest.by_id()
    ordered: list[Scene] = []
    seen: set[str] = set()
    current = by_id.get(manifest.start)
    while current is not None and current.id not in seen:
        ordered.append(current)
        seen.add(current.id)
        current = by_id.get(current.next)
    ordered.extend(s for s in manifest.scenes if s.id not in seen)
    return ordered


def set_video_transition(paths: RepoPaths, scene_id: str, video: str, duration: float) -> bool:
    """Point the scene's authored transition at the clip. Returns False when the YAML had to be
    re-dumped (comments lost); the line-level patch keeps them in every normal layout."""
    text = paths.scenes_yaml.read_text(encoding="utf-8")
    updates = {"type": "video", "video": video, "duration": duration}
    patched = set_scene_fields(text, scene_id, "transition", updates, TRANSITION_FIELD_ORDER)
    parse_manifest(yaml.safe_load(patched.text), paths.scenes_yaml)  # never write a bad file
    paths.scenes_yaml.write_text(patched.text, encoding="utf-8")
    return patched.in_place


def clip_in_place(paths: RepoPaths, scene: Scene) -> Path | None:
    """The clip the scene's authored transition plays, when it exists on disk."""
    if scene.transition.type != "video" or not scene.transition.video:
        return None
    clip = paths.public_file(scene.transition.video)
    return clip if clip.exists() else None


# --------------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------------


@dataclass
class PlannedEdge:
    a: str
    b: str
    mode: str
    """``veo`` (a paid call), or ``reencode`` (rebuild from the saved download, free)."""
    prompt: str
    first: ReferenceImage
    last: ReferenceImage
    request: dict[str, Any]
    estimate: float
    source_row: dict[str, Any] | None = None
    """For ``reencode``: the log row of the paid call that produced the saved download."""

    @property
    def what(self) -> str:
        return f"{self.a} -> {self.b}"


@dataclass
class MorphPlan:
    items: list[PlannedEdge] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def estimate(self) -> float:
        return sum(item.estimate for item in self.items)

    @property
    def paid(self) -> list[PlannedEdge]:
        return [item for item in self.items if item.mode == "veo"]


def pick_edges(
    manifest: Manifest, edge: tuple[str, str] | None, all_edges: bool
) -> list[tuple[str, str]]:
    """The edges a run works on; raises ValueError for a bad choice."""
    if edge and all_edges:
        raise ValueError("pass either --edge FROM TO or --all, not both")
    if all_edges:
        return [(s.id, s.next) for s in loop_order(manifest)]
    if not edge:
        raise ValueError("pass --edge FROM TO or --all")
    a, b = edge
    by_id = manifest.by_id()
    for sid in (a, b):
        if sid not in by_id:
            raise ValueError(f"unknown scene '{sid}'; known scenes: {', '.join(by_id)}")
    if by_id[a].next != b:
        raise ValueError(
            f"'{b}' is not the next scene of '{a}' (that is '{by_id[a].next}'); a morph clip "
            "plays on the edge from a scene to its next scene"
        )
    return [(a, b)]


def _references_log(paths: RepoPaths, first: ReferenceImage, last: ReferenceImage) -> list[Any]:
    return [
        {"role": role, "path": _rel(paths, ref.path), "sha256": ref.sha256, "bytes": len(ref.data)}
        for role, ref in (("first_frame", first), ("last_frame", last))
    ]


def _parameters(request: dict[str, Any]) -> dict[str, Any]:
    """The request minus the prompt and the images, which the log row stores separately."""
    config = {k: v for k, v in request.get("config", {}).items() if k != "last_frame"}
    return {"endpoint": request.get("endpoint"), **config}


def resumable_row(
    rows: list[dict[str, Any]], paths: RepoPaths, item: PlannedEdge
) -> dict[str, Any] | None:
    """The paid call whose saved download never became a clip, if it matches this request.

    Only the latest paid call for the edge counts, and only when its post-processing did not
    finish: a clip that was finished and later removed on purpose (a rejected clip) is never
    brought back for free behind a person's back.
    """
    raw = raw_path(paths, item.a, item.b)
    if not raw.exists():
        return None
    latest = None
    for row in rows:
        if (
            row.get("kind") == "morph"
            and row.get("purpose") == "morph"
            and row.get("scene") == item.a
            and row.get("next_scene") == item.b
        ):
            latest = row
    if latest is None or latest.get("status") != "ok" or latest.get("output_path"):
        return None
    if latest.get("raw_sha256") != sha256_bytes(raw.read_bytes()):
        return None
    same_request = (
        latest.get("prompt") == item.prompt
        and latest.get("model") == item.request.get("model")
        and latest.get("parameters") == _parameters(item.request)
        and [r.get("sha256") for r in latest.get("references", [])]
        == [item.first.sha256, item.last.sha256]
    )
    return latest if same_request else None


def source_row_for_raw(
    rows: list[dict[str, Any]], paths: RepoPaths, a: str, b: str
) -> dict[str, Any] | None:
    """The paid call that produced the saved download of this edge, for ``--reencode``."""
    raw = raw_path(paths, a, b)
    if not raw.exists():
        return None
    digest = sha256_bytes(raw.read_bytes())
    for row in reversed(rows):
        if (
            row.get("kind") == "morph"
            and row.get("purpose") == "morph"
            and row.get("raw_sha256") == digest
        ):
            return row
    return None


def plan_morphs(
    paths: RepoPaths,
    manifest: Manifest,
    edges: list[tuple[str, str]],
    veo: VeoProvider,
    *,
    style: StyleBlock,
    resolution: str,
    duration: int,
    force: bool = False,
    reencode: bool = False,
    dry_run: bool = False,
    rows: list[dict[str, Any]] | None = None,
) -> MorphPlan:
    plan = MorphPlan()
    rows = rows if rows is not None else genlog.read_rows(paths.generation_log)
    by_id = manifest.by_id()
    veo.check(resolution, duration)
    for a, b in edges:
        origin, destination = by_id[a], by_id[b]
        if reencode:
            source = source_row_for_raw(rows, paths, a, b)
            if source is None:
                plan.problems.append(
                    f"{a} -> {b}: no saved Veo download to re-encode "
                    f"({_rel(paths, raw_path(paths, a, b))} is missing or not in the log)"
                )
                continue
        elif not force and clip_in_place(paths, origin) is not None:
            plan.skipped.append(f"{a} -> {b}")
            continue
        missing = [
            _rel(paths, paths.graded(sid)) for sid in (a, b) if not paths.graded(sid).exists()
        ]
        if missing:
            plan.problems.append(f"{a} -> {b}: {', '.join(missing)} missing; run `dsp grade`")
            continue
        placeholders = [sid for sid in (a, b) if is_placeholder_image(paths.master(sid))]
        if placeholders and not reencode:
            problem = f"{a} -> {b}: {', '.join(placeholders)} still has a placeholder master"
            (plan.notes if dry_run else plan.problems).append(problem)
            if not dry_run:
                continue
        first = ReferenceImage.from_path(paths.graded(a))
        last = ReferenceImage.from_path(paths.graded(b))
        prompt = build_morph_prompt(style.text, origin, destination)
        item = PlannedEdge(
            a=a,
            b=b,
            mode="veo",
            prompt=prompt,
            first=first,
            last=last,
            request=veo.request(prompt, first, last, resolution, duration),
            estimate=veo.estimate_cost(resolution, duration),
        )
        if reencode:
            item.mode, item.estimate, item.source_row = "reencode", 0.0, source
        elif not force:
            resumable = resumable_row(rows, paths, item)
            if resumable is not None:
                item.mode, item.estimate, item.source_row = "reencode", 0.0, resumable
                plan.notes.append(
                    f"{a} -> {b}: finishing the saved Veo download from log row "
                    f"{resumable.get('id')} (no new call)"
                )
        plan.items.append(item)
    return plan


# --------------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------------


@dataclass
class MorphOutcome:
    clips: list[EdgeReview] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    spent: float = 0.0
    manifest_changed: bool = False


def wait_for_operation(
    veo: VeoProvider,
    operation: Any,
    *,
    what: str,
    echo: Echo,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    interval: float = POLL_INTERVAL_S,
    timeout: float = OPERATION_TIMEOUT_S,
) -> tuple[Any, int]:
    """Poll until the operation is done; returns it with the number of polls made."""
    started = clock()
    polls = 0
    while not getattr(operation, "done", False):
        if clock() - started > timeout:
            raise ProviderError(
                f"veo is still rendering {what} after {timeout / 60:.0f} minutes "
                f"(operation {getattr(operation, 'name', '?')}); giving up on it",
                retryable=False,
            )
        sleep(interval)
        operation = with_retries(
            lambda op=operation: veo.refresh(op), what=f"{what} (poll)", echo=echo, sleep=sleep
        )
        polls += 1
        if polls % PROGRESS_EVERY_POLLS == 0 and not getattr(operation, "done", False):
            echo(f"  still rendering after {clock() - started:.0f} s ...")
    return operation, polls


def _finish_edge(
    paths: RepoPaths,
    runner: Runner,
    item: PlannedEdge,
    raw: Path,
    *,
    model: str | None,
    cost: float | None,
    log_id: str | None,
    echo: Echo,
) -> tuple[EncodedClip, EdgeReview, bool]:
    """Encode, review and point the manifest at the clip. Returns (clip, review, in_place)."""
    encoded = encode_clip(paths, runner, raw, item.a, item.b, echo)
    review = review_clip(
        paths, runner, item.a, item.b, encoded, model=model, cost=cost, log_id=log_id
    )
    in_place = set_video_transition(
        paths, item.a, f"morphs/{encoded.path.name}", encoded.info.duration
    )
    return encoded, review, in_place


def _output_fields(paths: RepoPaths, encoded: EncodedClip) -> dict[str, Any]:
    return {
        "output_path": _rel(paths, encoded.path),
        "output_sha256": encoded.sha256,
        "output_bytes": encoded.info.size,
        "output_info": {**encoded.info.as_dict(), "crf": encoded.crf},
    }


def _report(review: EdgeReview, echo: Echo) -> None:
    echo(
        f"  wrote web/public/{review.clip} ({review.duration} s, {review.width}x{review.height}, "
        f"{review.bytes / 1e6:.2f} MB); first frame vs {review.a}: {review.first_vs_a}, "
        f"last frame vs {review.b}: {review.last_vs_b}"
    )
    for warning in review.warnings():
        echo(f"  warning: {warning}; look at the frames before keeping this clip")


def execute_plan(
    paths: RepoPaths,
    veo: VeoProvider,
    plan: MorphPlan,
    runner: Runner,
    *,
    style: StyleBlock,
    guard: SpendGuard,
    echo: Echo = print,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> MorphOutcome:
    """Render every planned edge, one after another; a failed edge does not stop the others,
    a spend-limit stop ends the run."""
    outcome = MorphOutcome()
    for item in plan.items:
        raw = raw_path(paths, item.a, item.b)
        if item.mode == "reencode":
            echo(f"re-encoding {item.what} from {_rel(paths, raw)} (no new call) ...")
            source = item.source_row or {}
            row: dict[str, Any] = {
                "id": genlog.new_row_id(),
                "timestamp": genlog.utc_now(),
                "kind": "morph",
                "purpose": "reencode",
                "provider": "ffmpeg",
                "model": source.get("model"),
                "scene": item.a,
                "next_scene": item.b,
                "source_row": source.get("id"),
                "raw_path": _rel(paths, raw),
                "raw_sha256": source.get("raw_sha256"),
                "cost_estimate_usd": 0.0,
                "cost_usd": 0.0,
            }
            started = time.monotonic()
            try:
                encoded, review, in_place = _finish_edge(
                    paths,
                    runner,
                    item,
                    raw,
                    model=source.get("model"),
                    cost=source.get("cost_usd"),
                    log_id=source.get("id"),
                    echo=echo,
                )
            except (ToolError, OSError, ValueError, PatchError) as error:
                row.update(status="error", error=str(error))
                echo(f"  failed: {error}")
                outcome.failed.append(item.what)
            else:
                row.update(status="ok", error=None, **_output_fields(paths, encoded))
                outcome.clips.append(review)
                outcome.manifest_changed = True
                if not in_place:
                    echo("  note: content/scenes.yaml was re-dumped; its comments were lost")
                _report(review, echo)
            row["duration_s"] = round(time.monotonic() - started, 2)
            genlog.append_row(paths.generation_log, row)
            continue

        guard.before_call(item.estimate, item.what)
        row = {
            "id": genlog.new_row_id(),
            "timestamp": genlog.utc_now(),
            "kind": "morph",
            "purpose": "morph",
            "provider": veo.name,
            "model": veo.model,
            "scene": item.a,
            "next_scene": item.b,
            "prompt": item.prompt,
            "parameters": _parameters(item.request),
            "references": _references_log(paths, item.first, item.last),
            "seed": None,
            "style_block_sha256": style.sha256,
            "cost_estimate_usd": item.estimate,
        }
        started = time.monotonic()
        echo(f"rendering {item.what} with {veo.name} {veo.model} (about ${item.estimate:.2f}) ...")
        try:
            operation = with_retries(
                lambda item=item: veo.start(
                    item.prompt,
                    item.first,
                    item.last,
                    item.request["config"]["resolution"],
                    item.request["config"]["duration_seconds"],
                ),
                what=item.what,
                echo=echo,
                sleep=sleep,
            )
            row["response_id"] = getattr(operation, "name", None)
            operation, polls = wait_for_operation(
                veo, operation, what=item.what, echo=echo, sleep=sleep, clock=clock
            )
            row["polls"] = polls
            video = veo.finished_video(operation)
            data = with_retries(
                lambda video=video: veo.download(video),
                what=f"{item.what} (download)",
                echo=echo,
                sleep=sleep,
            )
        except MissingCredentials:
            raise  # nothing was sent; the run stops
        except ProviderError as error:
            row.update(
                status="error",
                error=str(error),
                cost_usd=error.cost_usd,
                duration_s=round(time.monotonic() - started, 2),
            )
            genlog.append_row(paths.generation_log, row)
            if error.cost_usd:
                guard.record(f"{item.what} (failed)", error.cost_usd)
                outcome.spent += error.cost_usd
            echo(f"  failed: {error}")
            outcome.failed.append(item.what)
            continue

        # Paid for from here on: Veo bills per second of video, so the estimate is the cost.
        cost = item.estimate
        guard.record(item.what, cost)
        outcome.spent += cost
        row.update(
            status="ok",
            error=None,
            cost_usd=cost,
            raw_mime_type=video.mime_type,
            raw_path=_rel(paths, raw),
            raw_sha256=sha256_bytes(data),
            raw_bytes=len(data),
        )
        try:
            atomic_write(raw, data)
            row["raw_info"] = probe(runner, raw).as_dict()
            encoded, review, in_place = _finish_edge(
                paths, runner, item, raw, model=veo.model, cost=cost, log_id=row["id"], echo=echo
            )
        except (ToolError, OSError, ValueError, PatchError) as error:
            row["postprocess_error"] = str(error)
            echo(f"  post-processing failed: {error}")
            echo("  the download is kept; the next run finishes it without a new call")
            outcome.failed.append(item.what)
        else:
            row.update(_output_fields(paths, encoded))
            outcome.clips.append(review)
            outcome.manifest_changed = True
            if not in_place:
                echo("  note: content/scenes.yaml was re-dumped; its comments were lost")
            _report(review, echo)
        finally:
            row["duration_s"] = round(time.monotonic() - started, 2)
            genlog.append_row(paths.generation_log, row)
    return outcome


def print_requests(plan: MorphPlan, echo: Echo) -> None:
    for number, item in enumerate(plan.items, start=1):
        how = "re-encode the saved download" if item.mode == "reencode" else "Veo request"
        echo(
            f"edge {number} of {len(plan.items)}: {item.what} ({how}, "
            f"estimated ${item.estimate:.2f})"
        )
        if item.mode == "veo":
            echo(json.dumps(item.request, indent=2, ensure_ascii=False))


def run_morph(
    paths: RepoPaths,
    manifest: Manifest,
    edges: list[tuple[str, str]],
    veo: VeoProvider,
    *,
    resolution: str = DEFAULT_RESOLUTION,
    duration: int = DEFAULT_DURATION,
    force: bool = False,
    reencode: bool = False,
    dry_run: bool = False,
    guard: SpendGuard | None = None,
    runner: Runner | None = None,
    echo: Echo = print,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """The whole command; returns the exit code."""
    if force and reencode:
        echo("error: pass either --force (new Veo call) or --reencode (no call), not both")
        return 1
    try:
        style = load_style_block(paths, allow_draft=dry_run)
        plan = plan_morphs(
            paths,
            manifest,
            edges,
            veo,
            style=style,
            resolution=resolution,
            duration=duration,
            force=force,
            reencode=reencode,
            dry_run=dry_run,
        )
    except (StyleBlockError, ValueError) as error:
        echo(f"error: {error}")
        return 1

    if plan.skipped:
        echo(f"clip already in place, skipped (use --force to redo): {', '.join(plan.skipped)}")
    for note in plan.notes:
        echo(f"note: {note}")
    for problem in plan.problems:
        echo(("note: a real run would refuse: " if dry_run else "error: ") + problem)
    if plan.problems and not dry_run:
        return 1
    if not plan.items:
        echo("nothing to render")
        return 0

    paid = plan.paid
    description = f"{len(paid)} clip(s) of {duration} s at {resolution} with {veo.name} {veo.model}"
    guard = guard or SpendGuard.from_env()
    if dry_run:
        print_requests(plan, echo)
        if runner is None and shutil.which("ffmpeg") is None:
            echo("note: a real run would refuse: ffmpeg is not installed (`brew install ffmpeg`)")
        try:
            echo(guard.check_plan(plan.estimate, description))
        except SpendLimitExceeded as error:
            echo(f"note: a real run would refuse: {error}")
        echo("dry run: nothing was sent")
        return 0

    if runner is None:
        try:
            require_ffmpeg()
        except ToolError as error:
            echo(f"error: {error}")
            return 1
        runner = subprocess_runner
    try:
        echo(guard.check_plan(plan.estimate, description))
        outcome = execute_plan(
            paths,
            veo,
            plan,
            runner,
            style=style,
            guard=guard,
            echo=echo,
            sleep=sleep,
            clock=clock,
        )
    except (SpendLimitExceeded, MissingCredentials) as error:
        echo(f"error: {error}")
        return 1

    code = 1 if outcome.failed else 0
    if outcome.manifest_changed:
        code = max(code, _export(paths, echo))
    try:
        from .manifest import load_manifest

        index = write_review_index(paths, load_manifest(paths.scenes_yaml))
        echo(f"review sheet: {_rel(paths, index)}")
    except (ManifestError, OSError) as error:
        echo(f"warning: could not write the review sheet: {error}")
    echo(
        f"rendered {len(outcome.clips)} clip(s), {len(outcome.failed)} failed; "
        f"spent about ${outcome.spent:.2f}"
    )
    echo(genlog.spend_summary(paths.generation_log))
    return code


def _export(paths: RepoPaths, echo: Echo) -> int:
    """Re-export ``web/public/scenes.json`` so the site sees the new transitions."""
    from .export import run_export

    echo("re-exporting web/public/scenes.json ...")
    try:
        report = run_export(paths)
    except ManifestError as error:
        echo(f"error: {error}")
        return 1
    for problem in report.errors:
        echo(f"error: {problem}")
    if report.errors:
        echo("error: scenes.json was not updated; fix the above and run `dsp export`")
        return 1
    echo(
        "web/public/scenes.json is unchanged"
        if report.scenes_json == "unchanged"
        else "wrote web/public/scenes.json"
    )
    return 0
