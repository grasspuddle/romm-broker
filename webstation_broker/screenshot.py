"""Save-state frame capture over the pixelflux computer-use endpoint.

Selkies serves the Anthropic Computer Use HTTP spec on a loopback port when
`PIXELFLUX_CU` names one, and its `screenshot` action answers with the whole
framebuffer as a base64 PNG. The broker shares the container, so the port is
fixed rather than configured. The frame is scaled down here to the thumbnail
RomM files beside a state.
"""

import base64
import io
import logging
from typing import Optional

import httpx
from PIL import Image, UnidentifiedImageError

from . import settings

log = logging.getLogger(__name__)

CU_URL = "http://127.0.0.1:5000/computer-use"
"""The computer-use endpoint selkies binds inside the container."""

CU_TIMEOUT = 5.0
"""Seconds a screenshot request gets; on Wayland it forces a one-frame GPU readback."""


def capture_frame() -> Optional[bytes]:
    """Grab the streamed desktop as it is right now, scaled for a state thumbnail.

    Never raises: the frame is taken on the way into a save, and a missing
    preview must not read as a failed save.

    Returns:
        PNG bytes bounded to `STATE_SCREENSHOT_SIZE` on the longest side, or
        None when the endpoint is unreachable or answers with anything that
        does not decode as an image.
    """
    try:
        response = httpx.post(CU_URL, json={"action": "screenshot"}, timeout=CU_TIMEOUT)
        response.raise_for_status()
        raw = base64.b64decode(response.json()["data"], validate=True)
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        log.warning("state frame capture failed: %s", exc)
        return None
    return _thumbnail(raw)


def _thumbnail(raw: bytes) -> Optional[bytes]:
    """Scale a captured frame to the thumbnail size and re-encode it as PNG.

    Args:
        raw: The full-size frame as the endpoint returned it.

    Returns:
        PNG bytes bounded to `STATE_SCREENSHOT_SIZE` on the longest side, or
        None when `raw` is not a decodable image.
    """
    size = settings.STATE_SCREENSHOT_SIZE
    try:
        with Image.open(io.BytesIO(raw)) as image:
            frame = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        log.warning("state frame is not a decodable image: %s", exc)
        return None
    frame.thumbnail((size, size))
    out = io.BytesIO()
    frame.save(out, format="PNG", optimize=True)
    log.debug("state frame captured: %dx%d, %d bytes", frame.width, frame.height, out.tell())
    return out.getvalue()
