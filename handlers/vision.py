"""Vision handler — preprocesses images via Nemotron Nano VL before sending to LLM.

Mirrors the voice pipeline pattern: Audio → Whisper → text, Image → Vision → text.
Uses the OpenAI-compatible vision API on the local vLLM instance (GPU 1).
"""

import base64
import io
import logging

import aiohttp
from PIL import Image

from config import VISION_URL, VISION_MODEL, VISION_ENABLED

log = logging.getLogger("nexus")

# Preprocessing limits
MAX_EDGE = 1024
JPEG_QUALITY = 85

# Default vision prompt — factual, detailed description
DEFAULT_PROMPT = (
    "Describe this image in detail. Include any visible text, labels, or captions "
    "(transcribe exactly). If it's a screenshot, document, chart, or diagram, "
    "describe its content and structure."
)


def preprocess_image(image_bytes: bytes) -> bytes:
    """Resize to max 1024px longest edge, convert to JPEG q85."""
    img = Image.open(io.BytesIO(image_bytes))

    # Convert RGBA/palette to RGB for JPEG
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")

    # Resize if needed
    w, h = img.size
    if max(w, h) > MAX_EDGE:
        scale = MAX_EDGE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


async def describe_image(image_bytes: bytes, custom_prompt: str | None = None) -> str:
    """Send image to vision model and return text description.

    Args:
        image_bytes: Raw image bytes (any format Pillow can read).
        custom_prompt: Override the default description prompt.

    Returns:
        Text description of the image.

    Raises:
        RuntimeError: If vision is disabled or the API call fails.
    """
    if not VISION_ENABLED:
        raise RuntimeError("Vision is not enabled")

    # Preprocess and encode
    processed = preprocess_image(image_bytes)
    b64 = base64.b64encode(processed).decode("ascii")

    prompt = custom_prompt or DEFAULT_PROMPT

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    url = f"{VISION_URL}/chat/completions"
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Vision API error {resp.status}: {body[:200]}")
            data = await resp.json()

    return data["choices"][0]["message"]["content"].strip()
