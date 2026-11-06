#!/usr/bin/env python3
from __future__ import annotations

"""Runtime-compatible entrypoint for the V4 multiview semantic oracle.

Transformers 5.19 accepts local image *paths* in multimodal chat content but does
not accept ``file://`` URIs there.  The original V4 source used ``Path.as_uri``
for both source-view and mask-adjudication images.  In Colab/Python 3.13 that
falls through to the base64 loader and fails with ``Incorrect padding``.

This entrypoint keeps the reviewed V4 court unchanged and normalizes only local
image references immediately before the processor consumes the messages.  The
normalization therefore applies uniformly to both localization and adjudication.
"""

from copy import deepcopy
from pathlib import Path
from urllib.parse import unquote, urlparse

from tools.research import run_multiview_semantic_oracle_v4 as v4


def normalize_local_image_sources_v1(messages):
    """Return a copy with ``file://`` image URIs converted to absolute paths."""

    out = deepcopy(messages)
    for message in out:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            value = item.get("image")
            if not isinstance(value, str) or not value.startswith("file://"):
                continue
            parsed = urlparse(value)
            if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
                raise RuntimeError(f"UNSUPPORTED_LOCAL_IMAGE_URI:{value}")
            local = Path(unquote(parsed.path)).resolve()
            item["image"] = str(local)
    return out


_original_qwen_generate = v4._qwen_generate


def _qwen_generate_r2(model, processor, messages, *, max_new_tokens: int) -> str:
    return _original_qwen_generate(
        model,
        processor,
        normalize_local_image_sources_v1(messages),
        max_new_tokens=max_new_tokens,
    )


v4._qwen_generate = _qwen_generate_r2


def main() -> int:
    return v4.main()


if __name__ == "__main__":
    raise SystemExit(main())
