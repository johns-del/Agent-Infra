#!/usr/bin/env python3
import argparse
import base64
import json
import re
import struct
import time
from urllib.request import ProxyHandler, Request, build_opener
import zlib


def color_image(rgb):
    def chunk(kind, data):
        checksum = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    size = 224
    pixels = (b"\x00" + bytes(rgb) * size) * size
    png = b"\x89PNG\r\n\x1a\n" + b"".join(
        [
            chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(pixels)),
            chunk(b"IEND", b""),
        ]
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:32782")
    parser.add_argument("--model", default="deepseek-v41-flash-nvfp4-t1")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    opener = build_opener(ProxyHandler({}))
    base = args.url.rstrip("/")
    with opener.open(base + "/model_info", timeout=args.timeout) as resp:
        model_info = json.load(resp)
    if model_info.get("has_image_understanding") is not True:
        raise RuntimeError("Server did not report image understanding")

    def check(name, content, expected, stream=False):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"thinking": False},
            "max_tokens": 32,
            "temperature": 0,
            "stream": stream,
        }
        request = Request(
            base + "/v1/chat/completions",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        started = time.monotonic()
        with opener.open(request, timeout=args.timeout) as response:
            status = response.status
            usage = None
            if stream:
                parts = []
                done = False
                reason = None
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    event = line[5:].strip()
                    if event == b"[DONE]":
                        done = True
                        break
                    for choice in json.loads(event).get("choices", []):
                        parts.append(choice.get("delta", {}).get("content") or "")
                        reason = choice.get("finish_reason") or reason
                if not done:
                    raise RuntimeError("SSE stream ended before [DONE]")
                answer = "".join(parts)
            else:
                result = json.load(response)
                choice = result["choices"][0]
                answer = choice["message"]["content"]
                reason = choice["finish_reason"]
                usage = result.get("usage")
        row = {
            "case": name,
            "http": status,
            "answer": answer,
            "finish_reason": reason,
            "seconds": round(time.monotonic() - started, 3),
        }
        if usage is not None:
            row["prompt_tokens"] = usage.get("prompt_tokens")
            row["completion_tokens"] = usage.get("completion_tokens")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if status != 200 or reason != "stop" or not re.fullmatch(
            rf"\s*{expected}[.!]?\s*", answer or "", re.IGNORECASE
        ):
            raise RuntimeError(f"{name} failed, expected {expected!r}")

    check("text-stream", "What is 17 + 25? Reply with only the number.", "42", stream=True)
    for color, rgb in [("red", (230, 15, 20)), ("blue", (20, 30, 230))]:
        check(
            f"image-{color}",
            [
                {"type": "image_url", "image_url": {"url": color_image(rgb)}},
                {"type": "text", "text": "Is this image red or blue? Answer with one word."},
            ],
            color,
        )


if __name__ == "__main__":
    main()
