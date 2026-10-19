"""`dsp morph` with a fake Veo client and a fake ffmpeg runner; never touches the network.

The tests near the end run the real ffmpeg on tiny generated clips when it is installed, and
are skipped otherwise.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from dreamscapes_pipeline import genlog, morph
from dreamscapes_pipeline.cli import app
from dreamscapes_pipeline.export import content_hash
from dreamscapes_pipeline.layout import MASTER_HEIGHT, MASTER_WIDTH, RepoPaths
from dreamscapes_pipeline.manifest import load_built_manifest, load_manifest
from dreamscapes_pipeline.providers.base import ProviderError
from dreamscapes_pipeline.providers.veo import VeoProvider
from dreamscapes_pipeline.spend import SpendGuard, SpendLimitExceeded
from tests.conftest import file_hashes, write_small_repo
from tests.fakes import offline, write_style  # noqa: F401

COLOURS = {
    "first-scene": (90, 120, 140),
    "second-scene": (200, 150, 100),
    "third-scene": (60, 60, 90),
}
EDGE = ("first-scene", "second-scene")
HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")


# --------------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------------


class HttpError(Exception):
    def __init__(self, code: int):
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeVeoClient:
    """Stands in for ``google.genai.Client``: ``models.generate_videos``, ``operations.get``
    and ``files.download``, with the operation finishing after ``polls_needed`` polls."""

    def __init__(
        self,
        video: bytes = b"raw veo clip with audio",
        polls_needed: int = 2,
        start_failures: list[Exception] | None = None,
        blocked: bool = False,
        download_failures: int = 0,
    ):
        self.video = video
        self.polls_needed = polls_needed
        self.start_failures = list(start_failures or [])
        self.blocked = blocked
        self.download_failures = download_failures
        self.starts: list[dict[str, Any]] = []
        self.polls = 0
        self.downloads = 0
        self.models = SimpleNamespace(generate_videos=self._generate)
        self.operations = SimpleNamespace(get=self._get)
        self.files = SimpleNamespace(download=self._download)

    def _generate(self, **kwargs: Any) -> Any:
        self.starts.append(kwargs)
        if self.start_failures:
            raise self.start_failures.pop(0)
        return SimpleNamespace(name=f"operations/{len(self.starts)}", done=False, error=None)

    def _get(self, operation: Any) -> Any:
        self.polls += 1
        if self.polls < self.polls_needed:
            return SimpleNamespace(name=operation.name, done=False, error=None, response=None)
        if self.blocked:
            response = SimpleNamespace(
                generated_videos=[],
                rai_media_filtered_count=1,
                rai_media_filtered_reasons=["looked like a person"],
            )
        else:
            video = SimpleNamespace(uri="https://example.invalid/v.mp4", mime_type="video/mp4")
            response = SimpleNamespace(generated_videos=[SimpleNamespace(video=video)])
        return SimpleNamespace(name=operation.name, done=True, error=None, response=response)

    def _download(self, *, file: Any) -> bytes:
        self.downloads += 1
        if self.download_failures:
            self.download_failures -= 1
            raise HttpError(503)
        return self.video


class FakeRunner:
    """Stands in for ffmpeg and ffprobe.

    Encodes write ``sizes[height]`` bytes (height None means "not scaled"); frame extraction
    writes a solid image in the colour ``frames[name]``; ffprobe describes whatever was written,
    and anything else as a 1080p clip with an audio track (a raw Veo download).
    """

    def __init__(
        self,
        sizes: dict[int | None, int] | None = None,
        frames: dict[str, tuple[int, int, int]] | None = None,
        fail_encodes: int = 0,
    ):
        self.sizes = sizes or {None: 1_500_000, 720: 900_000}
        self.frames = frames or {
            "first": COLOURS["first-scene"],
            "middle": (150, 140, 120),
            "last": COLOURS["second-scene"],
        }
        self.fail_encodes = fail_encodes
        self.calls: list[list[str]] = []
        self.written: dict[str, dict[str, Any]] = {}

    @property
    def encodes(self) -> list[list[str]]:
        return [c for c in self.calls if "-c:v" in c]

    def __call__(self, args: list[str]) -> str:
        self.calls.append(list(args))
        target = args[-1]
        if args[0] == "ffprobe":
            info = self.written.get(target, {"height": 1080, "audio": 1})
            height = info["height"]
            streams = [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": height * 16 // 9,
                    "height": height,
                    "pix_fmt": "yuv420p",
                    "nb_frames": "192",
                }
            ]
            streams += [{"codec_type": "audio", "codec_name": "aac"}] * info["audio"]
            return json.dumps({"streams": streams, "format": {"duration": "8.000000"}})
        if "-c:v" in args:
            if self.fail_encodes:
                self.fail_encodes -= 1
                raise morph.ToolError("ffmpeg exited with 1: simulated failure")
            height = None
            if "-vf" in args:
                height = int(args[args.index("-vf") + 1].split(":")[1])
            size = self.sizes[height]
            Path(target).write_bytes((b"7" if height else b"1") * size)
            self.written[target] = {"height": height or 1080, "audio": 0}
            return ""
        name = Path(target).stem
        Image.new("RGB", (192, 108), self.frames.get(name, (100, 100, 100))).save(target)
        return ""


# --------------------------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------------------------


def write_graded(paths: RepoPaths) -> None:
    for scene_id, colour in COLOURS.items():
        Image.new("RGB", (MASTER_WIDTH, MASTER_HEIGHT), colour).save(paths.graded(scene_id))


@pytest.fixture
def repo(tmp_path: Path) -> RepoPaths:
    """Three scenes in a loop with real (non-placeholder) masters, graded copies and a style."""
    paths = write_small_repo(tmp_path / "repo")
    write_graded(paths)
    write_style(paths, "A quiet painted place, softly lit.")
    return paths


def run(
    paths: RepoPaths,
    client: FakeVeoClient,
    runner: Any,
    *,
    edges: list[tuple[str, str]] | None = None,
    limit: float = 15.0,
    sleeps: list[float] | None = None,
    **options: Any,
) -> tuple[int, list[str]]:
    lines: list[str] = []
    manifest = load_manifest(paths.scenes_yaml)
    code = morph.run_morph(
        paths,
        manifest,
        edges or [EDGE],
        VeoProvider(client=client, root=paths.root),
        guard=SpendGuard(limit=limit),
        runner=runner,
        echo=lines.append,
        sleep=(sleeps.append if sleeps is not None else lambda _s: None),
        **options,
    )
    return code, lines


def morph_rows(paths: RepoPaths) -> list[dict[str, Any]]:
    return [r for r in genlog.read_rows(paths.generation_log) if r.get("kind") == "morph"]


# --------------------------------------------------------------------------------------------
# Prompt, edges and Veo settings
# --------------------------------------------------------------------------------------------


def test_prompt_is_style_block_then_motion(repo):
    manifest = load_manifest(repo.scenes_yaml)
    a, b = manifest.scene("first-scene"), manifest.scene("second-scene")
    b.alt = "A pond where a lantern floats."
    prompt = morph.build_morph_prompt("STYLE BLOCK.", a, b)
    assert prompt == (
        "STYLE BLOCK.\n\nA slow dreamlike dissolve. The camera drifts gently toward the lantern; "
        "the scene softens into mist and light and becomes a pond where a lantern floats. No "
        "text, no people, calm, continuous motion."
    )


@pytest.mark.parametrize(
    ("alt", "phrase"),
    [
        ("A prison yard at night.", "a prison yard at night"),
        ("At dawn a fleet of balloons.", "at dawn a fleet of balloons"),
        ("NASA hangar in fog.", "NASA hangar in fog"),
    ],
)
def test_short_description_follows_becomes(repo, alt, phrase):
    scene = load_manifest(repo.scenes_yaml).scene("first-scene")
    scene.alt = alt
    assert morph.short_description(scene) == phrase


def test_edges_follow_the_loop_and_must_be_real(repo):
    manifest = load_manifest(repo.scenes_yaml)
    assert morph.pick_edges(manifest, None, True) == [
        ("first-scene", "second-scene"),
        ("second-scene", "third-scene"),
        ("third-scene", "first-scene"),
    ]
    assert morph.pick_edges(manifest, EDGE, False) == [EDGE]
    with pytest.raises(ValueError, match="not the next scene"):
        morph.pick_edges(manifest, ("first-scene", "third-scene"), False)
    with pytest.raises(ValueError, match="unknown scene"):
        morph.pick_edges(manifest, ("first-scene", "nowhere"), False)
    with pytest.raises(ValueError, match="not both"):
        morph.pick_edges(manifest, EDGE, True)


def test_veo_settings_and_prices():
    veo = VeoProvider()
    assert veo.model == "veo-3.1-fast-generate-preview"
    assert veo.estimate_cost("1080p", 8) == pytest.approx(0.96)
    assert veo.estimate_cost("720p", 6) == pytest.approx(0.60)
    veo.check("720p", 4)
    with pytest.raises(ValueError, match="1080p needs 8-second"):
        veo.check("1080p", 6)
    with pytest.raises(ValueError, match="unsupported duration"):
        veo.check("720p", 5)
    with pytest.raises(ValueError, match="unsupported resolution"):
        veo.check("480p", 8)
    # An unknown model is priced like the most expensive option.
    assert VeoProvider(model="veo-9").estimate_cost("1080p", 8) == pytest.approx(4.8)


# --------------------------------------------------------------------------------------------
# The command with fakes
# --------------------------------------------------------------------------------------------


def test_dry_run_prints_the_request_and_changes_nothing(repo):
    client, runner = FakeVeoClient(), FakeRunner()
    before = file_hashes(repo.root)
    code, lines = run(repo, client, runner, dry_run=True)
    assert code == 0
    assert file_hashes(repo.root) == before
    assert client.starts == [] and runner.calls == []
    out = "\n".join(lines)
    assert '"endpoint": "client.models.generate_videos"' in out
    assert '"model": "veo-3.1-fast-generate-preview"' in out
    assert '"last_frame": "<image/png assets/masters/second-scene.graded.png' in out
    assert '"resolution": "1080p"' in out and '"duration_seconds": 8' in out
    assert "estimated spend: $0.96" in out
    assert "dry run: nothing was sent" in out


def test_render_writes_clip_manifest_export_log_and_review(repo):
    client, runner, sleeps = FakeVeoClient(polls_needed=3), FakeRunner(), []
    code, lines = run(repo, client, runner, sleeps=sleeps)
    assert code == 0, lines

    # The request: first frame A, last frame B, 16:9, 1080p, 8 s.
    (start,) = client.starts
    assert start["model"] == "veo-3.1-fast-generate-preview"
    assert start["image"].image_bytes == repo.graded("first-scene").read_bytes()
    assert start["image"].mime_type == "image/png"
    config = start["config"]
    assert config.last_frame.image_bytes == repo.graded("second-scene").read_bytes()
    assert (config.aspect_ratio, config.resolution, config.duration_seconds) == ("16:9", "1080p", 8)
    assert start["prompt"].startswith("A quiet painted place, softly lit.\n\nA slow dreamlike")
    assert client.polls == 3 and sleeps == [10.0, 10.0, 10.0] and client.downloads == 1

    # The raw download is kept; the encode drops audio and makes web-friendly H.264.
    raw = morph.raw_path(repo, *EDGE)
    assert raw.read_bytes() == client.video
    (encode,) = runner.encodes
    for flag in ("-an", "libx264", "yuv420p", "+faststart"):
        assert flag in encode
    assert encode[encode.index("-crf") + 1] == "23" and "-vf" not in encode

    # The clip is named after its bytes, and both manifests point at it.
    (clip,) = morph.edge_clips(repo, *EDGE)
    assert clip.name == f"first-scene__second-scene.{content_hash(clip.read_bytes())}.mp4"
    transition = load_manifest(repo.scenes_yaml).scene("first-scene").transition
    assert (transition.type, transition.video, transition.duration) == (
        "video",
        f"morphs/{clip.name}",
        8.0,
    )
    built = {s.id: s for s in load_built_manifest(repo.scenes_json).scenes}
    assert built["first-scene"].transition.model_dump() == {
        "type": "video",
        "video": f"morphs/{clip.name}",
        "duration": 8.0,
    }
    assert built["second-scene"].transition.type == "zoom"

    # One paid row in the log, with cost and provenance.
    (row,) = morph_rows(repo)
    assert row["purpose"] == "morph" and row["status"] == "ok"
    assert row["scene"] == "first-scene" and row["next_scene"] == "second-scene"
    assert row["cost_usd"] == pytest.approx(0.96)
    assert row["response_id"] == "operations/1" and row["polls"] == 3
    assert [r["role"] for r in row["references"]] == ["first_frame", "last_frame"]
    assert row["parameters"] == {
        "endpoint": "client.models.generate_videos",
        "aspect_ratio": "16:9",
        "resolution": "1080p",
        "duration_seconds": 8,
    }
    assert row["raw_info"]["audio_streams"] == 1
    assert row["output_path"] == f"web/public/morphs/{clip.name}"
    assert row["output_info"]["audio_streams"] == 0

    # Review: three frames, scores against the masters, and the index page.
    folder = morph.review_dir(repo, *EDGE)
    for name in ("first.png", "middle.png", "last.png", "strip.jpg"):
        assert (folder / name).exists()
    review = json.loads((folder / "review.json").read_text())
    assert review["first_vs_a"] == 0.0 and review["last_vs_b"] == 0.0
    assert review["first_vs_b"] > 50
    index = (morph.review_root(repo) / "index.html").read_text()
    assert "first-scene &rarr; second-scene" in index and "in the manifest" in index
    assert any("spent about $0.96" in line for line in lines)


def test_a_clip_in_place_is_skipped_unless_forced(repo):
    run(repo, FakeVeoClient(), FakeRunner())
    client = FakeVeoClient()
    code, lines = run(repo, client, FakeRunner())
    assert code == 0 and client.starts == []
    assert any("skipped" in line for line in lines)

    forced = FakeVeoClient(video=b"a second take")
    code, _ = run(repo, forced, FakeRunner(sizes={None: 1_400_000, 720: 800_000}), force=True)
    assert code == 0 and len(forced.starts) == 1
    # The old clip of the edge is gone; only the new one remains.
    (clip,) = morph.edge_clips(repo, *EDGE)
    assert load_manifest(repo.scenes_yaml).scene("first-scene").transition.video == (
        f"morphs/{clip.name}"
    )
    assert len(morph_rows(repo)) == 2


def test_an_oversized_clip_is_scaled_to_720p_before_raising_crf(repo):
    runner = FakeRunner(sizes={None: 6_000_000, 720: 3_000_000})
    code, lines = run(repo, FakeVeoClient(), runner)
    assert code == 0, lines
    first, second = runner.encodes
    assert "-vf" not in first
    assert second[second.index("-vf") + 1] == "scale=-2:720:flags=lanczos"
    assert second[second.index("-crf") + 1] == "23"
    review = json.loads((morph.review_dir(repo, *EDGE) / "review.json").read_text())
    assert (review["height"], review["bytes"]) == (720, 3_000_000)


def test_a_720p_clip_is_not_scaled_and_only_its_crf_rises(repo):
    runner = FakeRunner(sizes={None: 7_000_000})
    runner.written["raw"] = {"height": 720, "audio": 1}
    original = runner.__call__

    def call(args: list[str]) -> str:
        # Describe the raw download as 720p, the way Veo returns a --resolution 720p clip.
        if args[0] == "ffprobe" and args[-1].endswith(".veo.mp4"):
            args = [*args[:-1], "raw"]
        return original(args)

    code, _ = run(repo, FakeVeoClient(), call, resolution="720p")
    assert code == 1
    assert [e[e.index("-crf") + 1] for e in runner.encodes] == ["23", "26", "28"]
    assert all("-vf" not in e for e in runner.encodes)


def test_comments_in_the_manifest_survive(repo):
    text = repo.scenes_yaml.read_text()
    repo.scenes_yaml.write_text("# The loop, written by hand.\n" + text)
    assert run(repo, FakeVeoClient(), FakeRunner())[0] == 0
    patched = repo.scenes_yaml.read_text()
    assert patched.startswith("# The loop, written by hand.\n")
    assert "    transition:\n      type: video\n      video: morphs/first-scene__second-scene." in (
        patched
    )


def test_a_clip_that_never_fits_fails_without_touching_the_manifest(repo):
    runner = FakeRunner(sizes={None: 9_000_000, 720: 6_000_000})
    code, lines = run(repo, FakeVeoClient(), runner)
    assert code == 1
    # 1080p at CRF 23, then 720p at CRF 23, 26 and 28.
    assert [e[e.index("-crf") + 1] for e in runner.encodes] == ["23", "23", "26", "28"]
    assert any("could not get first-scene__second-scene under 5 MB" in line for line in lines)
    assert load_manifest(repo.scenes_yaml).scene("first-scene").transition.type == "zoom"
    assert morph.edge_clips(repo, *EDGE) == []


def test_the_spend_guard_stops_before_any_call(repo):
    client = FakeVeoClient()
    code, lines = run(repo, client, FakeRunner(), limit=0.5)
    assert code == 1 and client.starts == []
    assert any("refusing to start" in line for line in lines)
    assert morph_rows(repo) == []


def test_rate_limits_are_retried(repo):
    client = FakeVeoClient(start_failures=[HttpError(429)])
    sleeps: list[float] = []
    code, _ = run(repo, client, FakeRunner(), sleeps=sleeps)
    assert code == 0 and len(client.starts) == 2
    assert sleeps[0] == 2.0  # the first backoff, before the polls


def test_a_blocked_video_is_logged_free_and_keeps_the_zoom(repo):
    client = FakeVeoClient(blocked=True)
    code, lines = run(repo, client, FakeRunner())
    assert code == 1
    (row,) = morph_rows(repo)
    assert row["status"] == "error" and row["cost_usd"] == 0.0
    assert "blocked" in row["error"]
    assert load_manifest(repo.scenes_yaml).scene("first-scene").transition.type == "zoom"
    assert any("spent about $0.00" in line for line in lines)


def test_failed_post_processing_is_finished_later_without_paying_again(repo):
    code, _ = run(repo, FakeVeoClient(), FakeRunner(fail_encodes=1))
    assert code == 1
    (paid,) = morph_rows(repo)
    assert paid["status"] == "ok" and "simulated failure" in paid["postprocess_error"]
    assert "output_path" not in paid
    assert morph.raw_path(repo, *EDGE).exists()

    client = FakeVeoClient()
    code, lines = run(repo, client, FakeRunner())
    assert code == 0, lines
    assert client.starts == []
    assert any("finishing the saved Veo download" in line for line in lines)
    resumed = morph_rows(repo)[-1]
    assert resumed["purpose"] == "reencode" and resumed["source_row"] == paid["id"]
    assert resumed["cost_usd"] == 0.0
    assert load_manifest(repo.scenes_yaml).scene("first-scene").transition.type == "video"


def test_a_removed_clip_is_rendered_again_not_quietly_restored(repo):
    """A clip a person rejected (manifest back to zoom, file deleted) must not come back for
    free from the saved download; asking for the edge again means a new call."""
    run(repo, FakeVeoClient(), FakeRunner())
    for clip in morph.edge_clips(repo, *EDGE):
        clip.unlink()
    text = repo.scenes_yaml.read_text()
    data = yaml.safe_load(text)
    data["scenes"][0]["transition"] = {"type": "zoom"}
    repo.scenes_yaml.write_text(yaml.safe_dump(data, sort_keys=False))

    # --all leaves the rejected edge alone, without paying for anything.
    client = FakeVeoClient(video=b"another take")
    all_edges = morph.pick_edges(load_manifest(repo.scenes_yaml), None, True)
    code, lines = run(repo, client, FakeRunner(), edges=all_edges, dry_run=True)
    assert any("keeps the zoom, left alone" in line for line in lines)
    assert not any("edge 1 of" in line and "first-scene -> second-scene" in line for line in lines)

    # Named with --edge, it is rendered again (a new paid call, never the old download).
    code, _ = run(repo, client, FakeRunner(), named=True)
    assert code == 0 and len(client.starts) == 1


def test_reencode_uses_the_saved_download(repo):
    code, lines = run(repo, FakeVeoClient(), FakeRunner(), reencode=True, named=True)
    assert code == 1
    assert any("no saved Veo download to re-encode" in line for line in lines)

    run(repo, FakeVeoClient(), FakeRunner())
    client = FakeVeoClient()
    code, lines = run(repo, client, FakeRunner(sizes={None: 1_000_000, 720: 1}), reencode=True)
    assert code == 0, lines
    assert client.starts == []
    assert morph_rows(repo)[-1]["purpose"] == "reencode"
    (clip,) = morph.edge_clips(repo, *EDGE)
    assert clip.stat().st_size == 1_000_000


def reject_edge(paths: RepoPaths) -> None:
    """What a person does to reject a clip: zoom in the manifest, clip deleted."""
    for clip in morph.edge_clips(paths, *EDGE):
        clip.unlink()
    data = yaml.safe_load(paths.scenes_yaml.read_text())
    data["scenes"][0]["transition"] = {"type": "zoom"}
    paths.scenes_yaml.write_text(yaml.safe_dump(data, sort_keys=False))


def test_reencode_all_leaves_a_rejected_edge_alone_unless_named(repo):
    run(repo, FakeVeoClient(), FakeRunner())
    reject_edge(repo)
    all_edges = morph.pick_edges(load_manifest(repo.scenes_yaml), None, True)

    code, lines = run(repo, FakeVeoClient(), FakeRunner(), edges=all_edges, reencode=True)
    out = "\n".join(lines)
    assert code == 0
    assert "keeps the zoom, left alone" in out and "first-scene -> second-scene" in out
    assert load_manifest(repo.scenes_yaml).scene("first-scene").transition.type == "zoom"
    assert morph.edge_clips(repo, *EDGE) == []

    code, lines = run(repo, FakeVeoClient(), FakeRunner(), reencode=True, dry_run=True)
    assert any("keeps the zoom, left alone" in line for line in lines)

    # Named explicitly, the saved download is brought back on purpose.
    code, lines = run(repo, FakeVeoClient(), FakeRunner(), reencode=True, named=True)
    assert code == 0, lines
    assert load_manifest(repo.scenes_yaml).scene("first-scene").transition.type == "video"


def test_a_clip_made_from_older_masters_is_rendered_again(repo):
    run(repo, FakeVeoClient(), FakeRunner())
    Image.new("RGB", (MASTER_WIDTH, MASTER_HEIGHT), (10, 200, 10)).save(repo.graded("second-scene"))

    client = FakeVeoClient()
    code, lines = run(repo, client, FakeRunner(), dry_run=True)
    assert any(
        "assets/masters/second-scene.graded.png changed since the clip in place" in line
        and "planning a new render" in line
        for line in lines
    )
    assert any("edge 1 of 1: first-scene -> second-scene (Veo request" in line for line in lines)

    code, lines = run(repo, client, FakeRunner(), keep_stale=True)
    assert code == 0 and client.starts == []
    assert any("kept because of --keep-stale" in line for line in lines)

    code, lines = run(repo, client, FakeRunner())
    assert code == 0 and len(client.starts) == 1
    assert client.starts[0]["config"].last_frame.image_bytes == (
        repo.graded("second-scene").read_bytes()
    )
    # The new clip matches the current masters, so the next run leaves it alone.
    code, lines = run(repo, client, FakeRunner())
    assert len(client.starts) == 1 and any("skipped" in line for line in lines)


def test_a_reencoded_clip_traces_back_to_its_paid_call(repo):
    run(repo, FakeVeoClient(), FakeRunner())
    run(repo, FakeVeoClient(), FakeRunner(sizes={None: 1_200_000, 720: 1}), reencode=True)
    rows = genlog.read_rows(repo.generation_log)
    (clip,) = morph.edge_clips(repo, *EDGE)
    source = morph.clip_source_row(rows, clip)
    assert source is not None and source["purpose"] == "morph"
    assert morph.changed_masters(rows, repo, *EDGE, clip) == []


def test_a_failed_download_is_charged_and_kept_retrievable(repo):
    client = FakeVeoClient(download_failures=10)
    code, lines = run(repo, client, FakeRunner())
    assert code == 1
    (row,) = morph_rows(repo)
    assert row["status"] == "error" and row["billing"] == "charged"
    assert row["cost_usd"] == pytest.approx(0.96)
    assert row["response_id"] == "operations/1" and row["retrievable_until"]
    assert genlog.total_spend(genlog.read_rows(repo.generation_log)) == pytest.approx(0.96)
    assert any("spent about $0.96" in line for line in lines)


def test_a_start_failure_costs_nothing(repo):
    client = FakeVeoClient(start_failures=[HttpError(400)])
    code, _ = run(repo, client, FakeRunner())
    assert code == 1
    (row,) = morph_rows(repo)
    assert row["billing"] == "none" and row["cost_usd"] is None
    assert genlog.total_spend(genlog.read_rows(repo.generation_log)) == 0.0


class StopOnSecondCall(SpendGuard):
    def before_call(self, estimate: float, what: str) -> None:
        if self.calls:
            raise SpendLimitExceeded(f"stopping before {what}: test limit")
        super().before_call(estimate, what)


@pytest.mark.parametrize("stop", ["spend limit", "missing key"])
def test_scenes_json_is_exported_even_when_the_run_stops_early(repo, stop):
    all_edges = morph.pick_edges(load_manifest(repo.scenes_yaml), None, True)
    lines: list[str] = []
    if stop == "spend limit":
        veo, guard = VeoProvider(client=FakeVeoClient(), root=repo.root), StopOnSecondCall(15)
    else:
        # The first edge finishes a saved download for free; the second needs Veo and no
        # GEMINI_API_KEY is set, so the run stops there.
        run(repo, FakeVeoClient(), FakeRunner(fail_encodes=1))
        veo, guard = VeoProvider(root=repo.root), SpendGuard(limit=15)
    code = morph.run_morph(
        repo,
        load_manifest(repo.scenes_yaml),
        all_edges,
        veo,
        guard=guard,
        runner=FakeRunner(),
        echo=lines.append,
        sleep=lambda _s: None,
    )
    assert code == 1
    assert any(line.startswith("stopped early:") for line in lines)
    built = {s.id: s for s in load_built_manifest(repo.scenes_json).scenes}
    assert built["first-scene"].transition.type == "video"
    assert built["second-scene"].transition.type == "zoom"
    assert (morph.review_root(repo) / "index.html").exists()


def test_encodes_are_written_outside_the_public_folder(repo):
    runner = FakeRunner()
    run(repo, FakeVeoClient(), runner)
    (encode,) = runner.encodes
    target = Path(encode[-1])
    assert target.parent == morph.encode_dir(repo)
    assert not target.is_relative_to(repo.web_public)
    assert sorted(p.name for p in repo.morphs.iterdir()) == [
        p.name for p in morph.edge_clips(repo, *EDGE)
    ]


def test_the_review_index_survives_a_deleted_clip(repo):
    run(repo, FakeVeoClient(), FakeRunner())
    reject_edge(repo)
    index = morph.write_review_index(repo, load_manifest(repo.scenes_yaml))
    text = index.read_text()
    assert "not in the manifest (zoom)" in text
    assert "no longer in web/public/morphs" in text and "<video" not in text


def test_placeholder_masters_are_refused(placeholder_repo):
    client = FakeVeoClient()
    write_style(placeholder_repo)
    code, lines = run(
        placeholder_repo, client, FakeRunner(), edges=[("cliff-cottage", "prison-yard")]
    )
    assert code == 1 and client.starts == []
    assert any("placeholder master" in line for line in lines)


def test_missing_ffmpeg_stops_before_paying(repo, monkeypatch):
    monkeypatch.setattr(morph.shutil, "which", lambda _name: None)
    client = FakeVeoClient()
    lines: list[str] = []
    code = morph.run_morph(
        repo,
        load_manifest(repo.scenes_yaml),
        [EDGE],
        VeoProvider(client=client, root=repo.root),
        guard=SpendGuard(limit=15),
        echo=lines.append,
    )
    assert code == 1 and client.starts == []
    assert any("brew install ffmpeg" in line for line in lines)


def test_operation_timeout_is_reported(repo):
    client = FakeVeoClient(polls_needed=10_000)
    ticks = iter(range(0, 10_000, 60))
    lines: list[str] = []
    code = morph.run_morph(
        repo,
        load_manifest(repo.scenes_yaml),
        [EDGE],
        VeoProvider(client=client, root=repo.root),
        guard=SpendGuard(limit=15),
        runner=FakeRunner(),
        echo=lines.append,
        sleep=lambda _s: None,
        clock=lambda: float(next(ticks)),
    )
    assert code == 1
    (row,) = morph_rows(repo)
    assert row["status"] == "error" and "still rendering" in row["error"]
    # Veo may still finish and bill the clip: counted, flagged, and fetchable by name.
    assert row["billing"] == "unknown" and row["cost_usd"] == pytest.approx(0.96)
    assert row["response_id"] == "operations/1"
    assert "operations/1" in row["error"] and row["retrievable_until"]
    assert genlog.total_spend(genlog.read_rows(repo.generation_log)) == pytest.approx(0.96)
    assert any("can still be fetched until" in line for line in lines)
    assert any("spent about $0.96" in line for line in lines)


def test_veo_operation_errors_are_not_retried():
    veo = VeoProvider(client=FakeVeoClient())
    failed = SimpleNamespace(done=True, error={"code": 3, "message": "bad frame"}, response=None)
    with pytest.raises(ProviderError, match="bad frame") as caught:
        veo.finished_video(failed)
    assert not caught.value.retryable


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------

cli = CliRunner()


def test_cli_dry_run_and_argument_errors(repo, monkeypatch):
    monkeypatch.chdir(repo.root)
    result = cli.invoke(app, ["morph", "--all", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "edge 3 of 3: third-scene -> first-scene" in result.output
    assert "estimated spend: $2.88" in result.output

    result = cli.invoke(app, ["morph", "--edge", "first-scene", "third-scene", "--dry-run"])
    assert result.exit_code == 1 and "not the next scene" in result.output
    result = cli.invoke(app, ["morph", "--all", "--resolution", "4k", "--dry-run"])
    assert result.exit_code == 1 and "720p or 1080p" in result.output
    result = cli.invoke(app, ["morph", "--all", "--duration", "6", "--dry-run"])
    assert result.exit_code == 1 and "1080p needs 8-second" in result.output
    result = cli.invoke(app, ["morph", "--all", "--force", "--reencode"])
    assert result.exit_code == 1 and "not both" in result.output


def test_cli_real_run_without_a_key_sends_nothing(repo, monkeypatch):
    monkeypatch.chdir(repo.root)
    monkeypatch.setattr(morph, "require_ffmpeg", lambda: None)
    result = cli.invoke(app, ["morph", "--edge", *EDGE])
    assert result.exit_code == 1
    assert "GEMINI_API_KEY is not set" in result.output
    assert morph_rows(repo) == []


# --------------------------------------------------------------------------------------------
# Real ffmpeg on tiny clips
# --------------------------------------------------------------------------------------------


def hex_colour(rgb: tuple[int, int, int]) -> str:
    return "0x" + "".join(f"{c:02x}" for c in rgb)


def make_clip(path: Path, first: tuple[int, int, int], last: tuple[int, int, int]) -> bytes:
    """Two seconds at 320x180: one second of each colour, with a sine-wave audio track, the
    way Veo always delivers audio."""
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={hex_colour(first)}:s=320x180:d=1:r=24",
            "-f",
            "lavfi",
            "-i",
            f"color=c={hex_colour(last)}:s=320x180:d=1:r=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "2:a",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv444p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )
    return path.read_bytes()


@needs_ffmpeg
def test_real_ffmpeg_end_to_end(repo, tmp_path):
    video = make_clip(tmp_path / "veo.mp4", COLOURS["first-scene"], COLOURS["second-scene"])
    code, lines = run(repo, FakeVeoClient(video=video), morph.subprocess_runner)
    assert code == 0, lines

    (clip,) = morph.edge_clips(repo, *EDGE)
    info = morph.probe(morph.subprocess_runner, clip)
    assert info.audio_streams == 0
    assert (info.codec, info.pix_fmt) == ("h264", "yuv420p")
    assert (info.width, info.height) == (320, 180)
    assert info.duration == pytest.approx(2.0, abs=0.05)
    data = clip.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat"), "faststart puts the index first"
    assert b"Lavf" not in data[:4096] or b"title" not in data[:4096]

    review = json.loads((morph.review_dir(repo, *EDGE) / "review.json").read_text())
    assert review["first_vs_a"] < 5 and review["last_vs_b"] < 5
    assert review["first_vs_b"] > 40 and review["last_vs_a"] > 40
    duration = load_manifest(repo.scenes_yaml).scene("first-scene").transition.duration
    assert duration == pytest.approx(2.0, abs=0.05)


@needs_ffmpeg
def test_real_frame_extraction_picks_first_middle_and_last(repo, tmp_path):
    clip = tmp_path / "clip.mp4"
    make_clip(clip, (255, 0, 0), (0, 0, 255))
    frames = morph.extract_frames(morph.subprocess_runner, clip, tmp_path / "frames", 2.0)
    first = Image.open(frames["first"]).convert("RGB").getpixel((160, 90))
    last = Image.open(frames["last"]).convert("RGB").getpixel((160, 90))
    assert first[0] > 200 and first[2] < 60
    assert last[2] > 200 and last[0] < 60
    with Image.open(frames["strip"]) as strip:
        assert strip.size == (4 * 480, 2 * 270)
