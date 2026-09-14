"""Capturing the streamed desktop for a state's thumbnail."""

import base64
import io
import json
from typing import Any

import httpx
import pytest
from PIL import Image

from webstation_broker import screenshot, settings


@pytest.fixture(autouse=True)
def no_frame_capture() -> None:
    """Override the shared stub: these tests drive the real capture against a fake endpoint."""


def _png(width: int, height: int) -> bytes:
    """Encode a solid image of the given size as PNG.

    Args:
        width: Pixel width.
        height: Pixel height.

    Returns:
        The PNG bytes.
    """
    out = io.BytesIO()
    Image.new("RGBA", (width, height), (10, 200, 30, 255)).save(out, format="PNG")
    return out.getvalue()


def _size(png: bytes) -> tuple[int, int]:
    """Read the pixel size back out of a PNG.

    Args:
        png: The image to measure.

    Returns:
        `(width, height)`.
    """
    with Image.open(io.BytesIO(png)) as image:
        return image.size


def _answer(monkeypatch: pytest.MonkeyPatch, body: Any, status: int = 200) -> list[dict[str, Any]]:
    """Make the computer-use endpoint answer `body`, recording every request.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        body: The JSON the endpoint answers with.
        status: The HTTP status it answers with.

    Returns:
        The list every request's `json` payload is appended to.
    """
    requests: list[dict[str, Any]] = []

    def post(url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
        requests.append(json)
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(screenshot.httpx, "post", post)
    return requests


def test_the_frame_is_scaled_to_the_thumbnail_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full-size capture comes back bounded on its longest side, aspect kept."""
    monkeypatch.setattr(settings, "STATE_SCREENSHOT_SIZE", 320)
    requests = _answer(monkeypatch, {"data": base64.b64encode(_png(1024, 768)).decode()})

    frame = screenshot.capture_frame()

    assert requests == [{"action": "screenshot"}]
    assert frame is not None
    assert frame.startswith(b"\x89PNG")
    assert _size(frame) == (320, 240)


def test_a_frame_already_small_enough_is_not_upscaled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The thumbnail size is a ceiling, not a target."""
    monkeypatch.setattr(settings, "STATE_SCREENSHOT_SIZE", 640)
    _answer(monkeypatch, {"data": base64.b64encode(_png(200, 100)).decode()})

    frame = screenshot.capture_frame()

    assert frame is not None
    assert _size(frame) == (200, 100)


def test_an_unreachable_endpoint_yields_no_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A container without the endpoint saves fine, just without a picture."""

    def refuse(url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(screenshot.httpx, "post", refuse)

    assert screenshot.capture_frame() is None


def test_an_error_status_yields_no_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """An endpoint that refuses the action is treated like one that is not there."""
    _answer(monkeypatch, {"error": "no backend"}, status=500)

    assert screenshot.capture_frame() is None


@pytest.mark.parametrize(
    "body",
    [{"text": "X=0,Y=0"}, {"data": "not base64!"}, {"data": base64.b64encode(b"not a png").decode()}],
)
def test_an_answer_that_is_not_an_image_yields_no_frame(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
) -> None:
    """Anything but a decodable image in `data` is dropped rather than served."""
    _answer(monkeypatch, body)

    assert screenshot.capture_frame() is None


def test_a_body_that_is_not_json_yields_no_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-JSON answer is dropped the same way."""

    def post(url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
        return httpx.Response(200, content=b"<html>", request=httpx.Request("POST", url))

    monkeypatch.setattr(screenshot.httpx, "post", post)

    assert screenshot.capture_frame() is None


def test_the_request_goes_to_the_loopback_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The broker shares the container with selkies, so the endpoint is the fixed loopback port."""
    seen: list[str] = []

    def post(url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
        seen.append(url)
        return httpx.Response(200, content=json_bytes({"data": ""}), request=httpx.Request("POST", url))

    def json_bytes(payload: dict[str, Any]) -> bytes:
        return json.dumps(payload).encode()

    monkeypatch.setattr(screenshot.httpx, "post", post)
    screenshot.capture_frame()

    assert seen == ["http://127.0.0.1:5000/computer-use"]
