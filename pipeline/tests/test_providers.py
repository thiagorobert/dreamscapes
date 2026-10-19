from __future__ import annotations

import base64

import httpx2
import openai
import pytest
from google.genai import errors as genai_errors

from dreamscapes_pipeline.providers import (
    FakeImageProvider,
    ImageProvider,
    MissingCredentials,
    ProviderError,
    ReferenceImage,
    make_image_provider,
)
from dreamscapes_pipeline.providers.base import synthetic_png
from dreamscapes_pipeline.providers.gemini import GeminiImageProvider
from dreamscapes_pipeline.providers.openai_images import OpenAIImageProvider, check_size
from tests.fakes import FakeGenaiClient, FakeOpenAIClient, offline  # noqa: F401


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "anchor.png"
    path.write_bytes(synthetic_png("anchor"))
    return ReferenceImage.from_path(path)


def test_every_provider_satisfies_the_protocol():
    assert isinstance(FakeImageProvider(), ImageProvider)
    assert isinstance(GeminiImageProvider(), ImageProvider)
    assert isinstance(OpenAIImageProvider(), ImageProvider)


def test_factory_defaults_and_errors():
    gemini = make_image_provider("gemini")
    assert (gemini.name, gemini.model, gemini.default_size()) == (
        "gemini",
        "gemini-nano-banana-2.1",
        "2K",
    )
    assert make_image_provider("").name == "gemini"
    openai_provider = make_image_provider("OpenAI", quality="low")
    assert (openai_provider.model, openai_provider.default_size()) == (
        "gpt-image-2.5-flare",
        "2048x1152",
    )
    with pytest.raises(ValueError, match="unknown image provider"):
        make_image_provider("flux")
    with pytest.raises(ValueError, match="only applies to the openai provider"):
        make_image_provider("gemini", quality="high")
    with pytest.raises(ValueError, match="quality must be one of"):
        make_image_provider("openai", quality="ultra")


# --------------------------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------------------------


def test_gemini_request_shape(reference):
    provider = GeminiImageProvider(root=reference.path.parent)
    request = provider.request("a prompt", "16:9", "2K", [reference], 7)
    assert request["endpoint"] == "client.interactions.create"
    assert request["model"] == "gemini-nano-banana-2.1"
    assert request["input"][0] == {"type": "text", "text": "a prompt"}
    image = request["input"][1]
    assert image["type"] == "image" and image["mime_type"] == "image/png"
    assert "anchor.png" in image["data"] and reference.sha256 in image["data"]
    assert request["response_format"] == {
        "type": "image",
        "aspect_ratio": "16:9",
        "image_size": "2K",
    }
    assert request["generation_config"] == {"seed": 7}
    assert "generation_config" not in provider.request("p", "16:9", "1K", [], None)


def test_gemini_rejects_unsupported_sizes():
    provider = GeminiImageProvider()
    with pytest.raises(ValueError, match="unsupported size"):
        provider.request("p", "16:9", "2048x1152", [], None)
    with pytest.raises(ValueError, match="unsupported size"):
        provider.request("p", "16:9", "2k", [], None)
    with pytest.raises(ValueError, match="unsupported aspect"):
        provider.request("p", "7:5", "2K", [], None)


def test_gemini_generate_sends_base64_references_and_reads_the_image(reference):
    client = FakeGenaiClient()
    provider = GeminiImageProvider(client=client)
    result = provider.generate("a prompt", "16:9", "2K", [reference], 11)
    sent = client.calls[0]
    assert "endpoint" not in sent
    assert base64.b64decode(sent["input"][1]["data"]) == reference.data
    assert sent["generation_config"] == {"seed": 11}
    assert result.image.startswith(b"\x89PNG")
    assert result.response_id == "interaction-123"
    assert result.seed == 11
    assert result.model == "gemini-nano-banana-2.1"
    # 1000 input tokens at $1.50/M, 1680 image tokens at $30/M, 400 thinking tokens at $7.50/M.
    assert result.cost_usd == pytest.approx((1000 * 1.5 + 1680 * 30 + 400 * 7.5) / 1e6)
    assert 0.05 < result.cost_estimate_usd < 0.08


def test_gemini_cost_estimates_follow_the_size():
    provider = GeminiImageProvider()
    one_k, two_k, four_k = (provider.estimate_cost(s, 0) for s in ("1K", "2K", "4K"))
    assert one_k < two_k < four_k
    assert provider.estimate_cost("2K", 1) > two_k


def test_gemini_without_an_image_is_not_retryable():
    def no_image(body):
        response = FakeGenaiClient.default_response(body)
        response.output_image = None
        response.output_text = "I can't draw that."
        return response

    provider = GeminiImageProvider(client=FakeGenaiClient(no_image))
    with pytest.raises(ProviderError, match="can't draw that") as caught:
        provider.generate("p", "16:9", "2K", [], None)
    assert not caught.value.retryable


@pytest.mark.parametrize(("code", "retryable"), [(429, True), (503, True), (400, False)])
def test_gemini_errors_are_classified(code, retryable):
    def fail(body):
        if code < 500:
            raise genai_errors.ClientError(code, {"error": {"message": "nope"}})
        raise genai_errors.ServerError(code, {"error": {"message": "busy"}})

    provider = GeminiImageProvider(client=FakeGenaiClient(fail))
    with pytest.raises(ProviderError) as caught:
        provider.generate("p", "16:9", "2K", [], None)
    assert caught.value.retryable is retryable
    assert caught.value.status_code == code


def test_gemini_needs_a_key():
    with pytest.raises(MissingCredentials, match="GEMINI_API_KEY"):
        GeminiImageProvider().generate("p", "16:9", "2K", [], None)


# --------------------------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "problem"),
    [
        ("2048x1150", "multiples of 16"),
        ("4096x2304", "exceed 3840"),
        ("1024x576", "total pixels"),
        ("3840x1024", "aspect ratio"),
        ("big", "look like"),
    ],
)
def test_openai_size_rules(size, problem):
    with pytest.raises(ValueError, match=problem):
        check_size(size)


def test_openai_size_aliases_and_aspect_check():
    provider = OpenAIImageProvider()
    assert provider.request("p", "16:9", "1K", [], None)["size"] == "1280x720"
    assert provider.request("p", "16:9", "4K", [], None)["size"] == "3840x2160"
    with pytest.raises(ValueError, match="aspect ratio"):
        provider.request("p", "16:9", "1024x1024", [], None)


def test_openai_without_references_uses_generate():
    client = FakeOpenAIClient()
    provider = OpenAIImageProvider(quality="medium", client=client)
    assert provider.request("p", "16:9", "2048x1152", [], 5)["endpoint"] == (
        "client.images.generate"
    )
    result = provider.generate("p", "16:9", "2048x1152", [], 5)
    kind, kwargs = client.calls[0]
    assert kind == "generate"
    assert kwargs == {
        "model": "gpt-image-2.5-flare",
        "prompt": "p",
        "size": "2048x1152",
        "quality": "medium",
        "output_format": "png",
        "n": 1,
    }
    assert result.seed is None, "the Image API has no seed"
    assert result.revised_prompt == "rewritten"
    assert result.request_id == "req_openai_1"
    assert result.cost_usd == pytest.approx((4000 * 30 + 500 * 8 + 200 * 5) / 1e6)


def test_openai_references_go_through_the_edits_endpoint(reference):
    client = FakeOpenAIClient()
    provider = OpenAIImageProvider(client=client, root=reference.path.parent)
    request = provider.request("p", "16:9", "2K", [reference], None)
    assert request["endpoint"] == "client.images.edit"
    assert "multipart image[] file" in request["image"][0]
    provider.generate("p", "16:9", "2K", [reference], None)
    kind, kwargs = client.calls[0]
    assert kind == "edit"
    assert kwargs["image"] == [("anchor.png", reference.data, "image/png")]
    assert kwargs["quality"] == "high"


def test_openai_estimates_grow_with_quality():
    costs = [OpenAIImageProvider(quality=q).estimate_cost("2K", 0) for q in ("low", "high", "max")]
    assert costs == sorted(costs) and costs[0] < 0.05 < costs[1]


def _status_error(cls, status, headers=None):
    request = httpx2.Request("POST", "https://api.example.invalid/v1/images")
    response = httpx2.Response(status, request=request, headers=headers or {})
    return cls("failed", response=response, body=None)


@pytest.mark.parametrize(
    ("error", "retryable", "retry_after"),
    [
        (_status_error(openai.RateLimitError, 429, {"retry-after": "7"}), True, 7.0),
        (_status_error(openai.InternalServerError, 500), True, None),
        (_status_error(openai.BadRequestError, 400), False, None),
        (
            openai.APIConnectionError(request=httpx2.Request("POST", "https://x.invalid")),
            True,
            None,
        ),
    ],
)
def test_openai_errors_are_classified(error, retryable, retry_after):
    provider = OpenAIImageProvider(client=FakeOpenAIClient(error=error))
    with pytest.raises(ProviderError) as caught:
        provider.generate("p", "16:9", "2K", [], None)
    assert caught.value.retryable is retryable
    assert caught.value.retry_after == retry_after


def test_openai_needs_a_key():
    with pytest.raises(MissingCredentials, match="OPENAI_API_KEY"):
        OpenAIImageProvider().generate("p", "16:9", "2K", [], None)


# --------------------------------------------------------------------------------------------
# Fake provider
# --------------------------------------------------------------------------------------------


def test_fake_provider_is_deterministic_and_records_calls():
    first, second = FakeImageProvider(), FakeImageProvider()
    a = first.generate("p", "16:9", "2K", [], 1)
    b = second.generate("p", "16:9", "2K", [], 1)
    assert a.image == b.image
    assert first.calls[0].seed == 1
    failing = FakeImageProvider(failures=[ProviderError("busy", retryable=True)])
    with pytest.raises(ProviderError):
        failing.generate("p", "16:9", "2K", [], 1)
    assert failing.generate("p", "16:9", "2K", [], 1).image
    assert failing.attempts == 2
