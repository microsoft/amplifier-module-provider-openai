"""Validate native computer screenshots before constructing an image envelope.

No text/error result is coerced into pixels. The original transcript stays
unchanged; an unsupported result is a local, non-retryable protocol failure.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from typing import Any

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000


def _dimensions(width: int, height: int) -> None:
    if width < 1 or height < 1 or width * height > MAX_PIXELS:
        raise ValueError("Screenshot dimensions are missing or exceed the pixel limit.")


def _png(data: bytes) -> None:
    offset, found_header, found_data = 8, False, False
    compressed = bytearray()
    while offset + 12 <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        end = offset + 12 + size
        if end > len(data):
            raise ValueError("Truncated PNG screenshot.")
        body = data[offset + 8 : offset + 8 + size]
        if zlib.crc32(kind + body) != int.from_bytes(
            data[offset + 8 + size : end], "big"
        ):
            raise ValueError("PNG screenshot checksum failed.")
        if not found_header:
            if kind != b"IHDR" or size != 13:
                raise ValueError("PNG screenshot is missing its header.")
            width, height, depth, color, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", body)
            )
            _dimensions(width, height)
            allowed = {
                0: {1, 2, 4, 8, 16},
                2: {8, 16},
                3: {1, 2, 4, 8},
                4: {8, 16},
                6: {8, 16},
            }
            if (
                depth not in allowed.get(color, set())
                or compression
                or filtering
                or interlace not in (0, 1)
            ):
                raise ValueError("Invalid PNG screenshot format.")
            found_header = True
        elif kind == b"IHDR":
            raise ValueError("Duplicate PNG screenshot header.")
        if kind == b"IDAT":
            compressed.extend(body)
            found_data = True
        if kind == b"IEND":
            if size or end != len(data) or not found_data:
                raise ValueError("Invalid PNG screenshot end.")
            # Bound decompression independently of an untrusted size header.
            limit = min(MAX_IMAGE_BYTES * 8, width * height * 8 + height * 8 + 1024)
            decoder = zlib.decompressobj()
            try:
                decoded = decoder.decompress(bytes(compressed), limit + 1)
            except zlib.error as exc:
                raise ValueError("Invalid PNG pixel stream.") from exc
            if (
                len(decoded) > limit
                or not decoder.eof
                or decoder.unconsumed_tail
                or decoder.unused_data
            ):
                raise ValueError("Invalid or oversized PNG pixel stream.")
            return
        offset = end
    raise ValueError("PNG screenshot is incomplete.")


def validate_image(data: bytes, mime: str) -> None:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Screenshot byte limit exceeded or image is empty.")
    if mime == "image/png" and data.startswith(b"\x89PNG\r\n\x1a\n"):
        _png(data)
        return
    if (
        mime == "image/jpeg"
        and data.startswith(b"\xff\xd8")
        and data.endswith(b"\xff\xd9")
    ):
        offset, dimensions = 2, False
        while offset + 4 <= len(data):
            if data[offset] != 255:
                break
            while offset < len(data) and data[offset] == 255:
                offset += 1
            marker = data[offset]
            offset += 1
            if marker == 0xDA:  # Scan data follows a checked frame header.
                if dimensions:
                    return
                break
            size = int.from_bytes(data[offset : offset + 2], "big")
            if size < 2 or offset + size > len(data):
                break
            if (
                marker
                in {
                    0xC0,
                    0xC1,
                    0xC2,
                    0xC3,
                    0xC5,
                    0xC6,
                    0xC7,
                    0xC9,
                    0xCA,
                    0xCB,
                    0xCD,
                    0xCE,
                    0xCF,
                }
                and size >= 8
            ):
                _dimensions(
                    int.from_bytes(data[offset + 5 : offset + 7], "big"),
                    int.from_bytes(data[offset + 3 : offset + 5], "big"),
                )
                dimensions = True
            offset += size
    elif (
        mime == "image/gif"
        and data[:6] in {b"GIF87a", b"GIF89a"}
        and len(data) >= 14
        and data[-1:] == b";"
        and b"," in data[13:]
    ):
        _dimensions(*struct.unpack("<HH", data[6:10]))
        return
    elif (
        mime == "image/webp"
        and len(data) >= 30
        and data[:4] == b"RIFF"
        and data[8:12] == b"WEBP"
        and int.from_bytes(data[4:8], "little") + 8 == len(data)
    ):
        kind, size = data[12:16], int.from_bytes(data[16:20], "little")
        if size + 20 <= len(data):
            if kind == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
                _dimensions(
                    int.from_bytes(data[26:28], "little") & 0x3FFF,
                    int.from_bytes(data[28:30], "little") & 0x3FFF,
                )
                return
            if kind == b"VP8L" and data[20] == 0x2F:
                value = int.from_bytes(data[21:25], "little")
                _dimensions((value & 0x3FFF) + 1, ((value >> 14) & 0x3FFF) + 1)
                return
            if kind == b"VP8X" and size == 10:
                _dimensions(
                    int.from_bytes(data[24:27], "little") + 1,
                    int.from_bytes(data[27:30], "little") + 1,
                )
                if b"VP8 " in data[30:] or b"VP8L" in data[30:]:
                    return
    raise ValueError(
        "Screenshot bytes do not match a supported image container and declared MIME type."
    )


def screenshot_data_url(content: Any) -> str:
    source = None
    if isinstance(content, str):
        if content.startswith("data:"):
            header, separator, encoded = content.partition(",")
            if not separator or not header.endswith(";base64"):
                raise ValueError("Screenshot data URL must contain base64 image bytes.")
            source = {"media_type": header[5:-7], "data": encoded}
        else:
            source = {"media_type": "image/png", "data": content}
    elif isinstance(content, list):
        if len(content) != 1:
            raise ValueError(
                "Native screenshot results cannot discard accompanying text or error content."
            )
        images = [
            block.get("source")
            for block in content
            if isinstance(block, dict) and block.get("type") == "image"
        ]
        if (
            len(images) == 1
            and isinstance(images[0], dict)
            and images[0].get("type") == "base64"
        ):
            source = images[0]
    if (
        not source
        or not isinstance(source.get("data"), str)
        or not source["data"]
        or len(source["data"]) > ((MAX_IMAGE_BYTES + 2) // 3) * 4
    ):
        raise ValueError(
            "computer_call tool result did not contain image data within the screenshot limit."
        )
    mime = source.get("media_type", "image/png")
    if mime not in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
        raise ValueError("Unsupported screenshot MIME type.")
    try:
        data = base64.b64decode(source["data"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Computer result is not base64 image data.") from exc
    validate_image(data, mime)
    return f"data:{mime};base64,{source['data']}"


def result_kind(content: Any) -> str:
    if isinstance(content, str):
        if len(content) > 64000:
            return "unvalidated_image"
        try:
            content = json.loads(content)
        except ValueError:
            return "text_or_invalid_image"
    if isinstance(content, dict):
        return (
            "error"
            if content.get("success") is False
            or content.get("isError") is True
            or content.get("error")
            else "structured_result"
        )
    return "content_blocks" if isinstance(content, list) else "unsupported_result"
