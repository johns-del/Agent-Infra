"""Optional SSE transport keepalive after native request validation/admission.

Comments are not model chunks or tokens. Keep one pending source read alive;
timing out a heartbeat must not cancel the underlying inference generator.
"""
import asyncio
from contextlib import suppress
import math
import os


def keepalive_seconds():
    value = float(os.environ.get('ORVIX_SSE_KEEPALIVE_SECONDS', '0'))
    return value if math.isfinite(value) and value > 0 else 0


async def admitted_keepalive(source, admitted, interval, encode_error):
    pending = None
    ready = asyncio.create_task(admitted.wait())
    published = False
    try:
        while True:
            pending = asyncio.create_task(source.__anext__())
            while not pending.done():
                if not admitted.is_set():
                    # No HTTP 200/heartbeat before native input validation.
                    await asyncio.wait({pending, ready}, return_when=asyncio.FIRST_COMPLETED)
                else:
                    done, _ = await asyncio.wait({pending}, timeout=interval)
                    if not done:
                        published = True
                        yield ': keep-alive\n\n'
            try:
                chunk = pending.result()
            except StopAsyncIteration:
                return
            except ValueError as exc:
                if not published:
                    raise
                # Once admitted SSE has started, report a later native error
                # using the existing OpenAI error envelope, never as success.
                yield 'data: ' + encode_error(str(exc)) + '\n\n'
                yield 'data: [DONE]\n\n'
                return
            published = True
            yield chunk
    finally:
        ready.cancel()
        with suppress(asyncio.CancelledError):
            await ready
        if pending is not None:
            if not pending.done():
                pending.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await pending
        await source.aclose()
