from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageDraw


DEFAULT_VENUS_ENDPOINT = "http://v2.open.venus.oa.com/llmproxy/chat/completions"
MAX_RETRY_BACKOFF_S = 45.0
RATE_LIMIT_JITTER_MAX_QPS = 1.0
# Reasoning-capable image models (e.g. gemini-3-pro-image) can emit several
# thousand reasoning tokens before the image itself.  The historical value of
# 1000 made these calls return empty choices after ~11.5k completion tokens.
DEFAULT_MAX_OUTPUT_TOKENS = 16384
RATE_LIMIT_JITTER_FRACTION = 0.35


class ProviderTransientResponseError(RuntimeError):
    """A syntactically valid provider response that did not deliver an image completion."""


def _empty_choices_diagnostic(response: Dict[str, Any]) -> str:
    venus_marker = response.get("venusMarker") if isinstance(response.get("venusMarker"), dict) else {}
    return (
        "Provider response contains no choices"
        f"; response_id={response.get('id')}"
        f"; model={response.get('model')}"
        f"; venus_span_id={venus_marker.get('spanId')}"
        f"; top_level_keys={sorted(response.keys())}"
        f"; finish_reason={_first_finish_reason(response)}"
        f"; usage={response.get('usage')}"
        f"; error={response.get('error')}"
        f"; venus_marker={sorted(venus_marker.keys())}"
    )


def _first_finish_reason(response: Dict[str, Any]) -> Any:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            return first.get("finish_reason") or first.get("finishReason")
    return None


def image_to_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _decode_image_value(value: str, timeout_s: int) -> Tuple[bytes, str]:
    if value.startswith("data:image/"):
        header, payload = value.split(",", 1)
        mime = header.split(";", 1)[0].split(":", 1)[1]
        return base64.b64decode(payload), mime
    if value.startswith("http://") or value.startswith("https://"):
        with urllib.request.urlopen(value, timeout=timeout_s) as response:
            return response.read(), response.headers.get_content_type()
    try:
        return base64.b64decode(value, validate=True), "image/png"
    except Exception as exc:
        raise ValueError("Unsupported image response value") from exc


def extract_image(response: Dict[str, Any], timeout_s: int) -> Tuple[bytes, str]:
    choices = response.get("choices") or []
    if not choices:
        raise ProviderTransientResponseError(_empty_choices_diagnostic(response))
    content = choices[0].get("message", {}).get("content", [])
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    for item in content:
        if not isinstance(item, dict):
            continue
        for key in ("venus_multimodal_url", "image_url"):
            block = item.get(key)
            if isinstance(block, dict) and block.get("url"):
                data, detected_mime = _decode_image_value(str(block["url"]), timeout_s)
                return data, str(block.get("mimeType") or block.get("mime_type") or detected_mime)
        for key in ("url", "base64", "b64_json"):
            if item.get(key):
                return _decode_image_value(str(item[key]), timeout_s)
    raise ValueError("Provider response contains no supported image item")


class ImageGenerationProvider(ABC):
    name: str
    model: str

    def generate_source(self, instruction: str, output_path: Path) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.name} does not support source text-to-image generation")

    @abstractmethod
    def edit_target(self, source_path: Path, instruction: str, output_path: Path) -> Dict[str, Any]:
        raise NotImplementedError


class MockImageProvider(ImageGenerationProvider):
    name = "mock"

    def __init__(self, model: str = "mock-local") -> None:
        self.model = model

    def edit_target(self, source_path: Path, instruction: str, output_path: Path) -> Dict[str, Any]:
        with Image.open(source_path) as image:
            target = image.convert("RGB")
        draw = ImageDraw.Draw(target)
        width, height = target.size
        box = (width // 3, height // 3, 2 * width // 3, 2 * height // 3)
        draw.rectangle(box, fill=(220, 70, 70), outline=(255, 255, 255), width=max(1, width // 128))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        target.save(output_path, format="PNG")
        return {"response_id": None, "mime_type": "image/png", "mock": True}


class NanoBananaProProvider(ImageGenerationProvider):
    name = "nano_banana_pro"

    def __init__(
        self,
        model: str = "gemini-3-pro-image",
        api_key_env: str = "CFCOT_IMAGE_API_KEY",
        endpoint: str = DEFAULT_VENUS_ENDPOINT,
        timeout_s: int = 120,
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
        rate_limit_qps: float = 0.2,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = int(timeout_s)
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)
        self.rate_limit_qps = float(rate_limit_qps)
        # Reasoning-capable image models can spend thousands of tokens thinking
        # before emitting the image; a small budget yields empty choices.
        self.max_output_tokens = int(max_output_tokens)
        self.api_key = "jDwO4Q8XZVko830MDqDx4stH@2198"
        if not self.api_key:
            raise ValueError(f"Missing API key in environment variable {api_key_env}")
        self._last_request_at = 0.0

    def _wait_for_rate_limit(self) -> None:
        if self.rate_limit_qps <= 0:
            return
        interval = 1.0 / self.rate_limit_qps
        wait_s = interval - (time.monotonic() - self._last_request_at)
        if wait_s > 0:
            time.sleep(wait_s)
        # Spread concurrent callers (and repeated batched runs) across the
        # interval so our own requests do not arrive in one burst.
        if self.rate_limit_qps <= RATE_LIMIT_JITTER_MAX_QPS:
            time.sleep(RATE_LIMIT_JITTER_FRACTION * interval)

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        encoded = json.dumps(payload).encode("utf-8")
        retryable_codes = {429, 500, 502, 503, 504}
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit()
            request = urllib.request.Request(
                self.endpoint,
                data=encoded,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    self._last_request_at = time.monotonic()
                    response_payload = json.loads(response.read().decode("utf-8"))
                    if not response_payload.get("choices"):
                        last_error = ProviderTransientResponseError(_empty_choices_diagnostic(response_payload))
                    else:
                        return response_payload
            except urllib.error.HTTPError as exc:
                self._last_request_at = time.monotonic()
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP Error {exc.code}: {exc.reason}; response_body={error_body}")
                if exc.code not in retryable_codes:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
            if attempt < self.max_retries:
                # Add bounded deterministic jitter so repeated empty completions do
                # not synchronize with a transiently overloaded upstream worker.
                # The exponential term is capped: upstream empty completions are the
                # dominant failure mode and a long sleep only delays a cheap retry.
                backoff_s = min(self.retry_backoff_s * (2 ** (attempt - 1)), MAX_RETRY_BACKOFF_S)
                time.sleep(backoff_s + min(1.0, 0.17 * attempt))
        assert last_error is not None
        raise last_error

    def generate_source(self, instruction: str, output_path: Path) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": instruction},
            ]}],
            "max_tokens": self.max_output_tokens,
        }
        response = self._post(payload)
        image_bytes, mime_type = extract_image(response, self.timeout_s)
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format="PNG")
        return {"response_id": response.get("id"), "mime_type": mime_type, "mock": False}

    def edit_target(self, source_path: Path, instruction: str, output_path: Path) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": image_to_data_url(source_path)}},
            ]}],
            "max_tokens": self.max_output_tokens,
        }
        response = self._post(payload)
        image_bytes, mime_type = extract_image(response, self.timeout_s)
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format="PNG")
        return {"response_id": response.get("id"), "mime_type": mime_type, "mock": False}


def build_provider(name: str, **kwargs: Any) -> ImageGenerationProvider:
    if name == "mock":
        return MockImageProvider(model=str(kwargs.get("model") or "mock-local"))
    if name == "nano_banana_pro":
        return NanoBananaProProvider(**kwargs)
    raise ValueError(f"Unsupported image provider: {name}")
