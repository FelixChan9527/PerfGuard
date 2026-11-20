from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import httpx
from openai import APIConnectionError

logger = logging.getLogger(__name__)


class LLMStreamInterruptedError(ConnectionError):
    pass

async def collect_completion(client, arguments, *, idle_timeout, total_timeout, metrics):
    started = last_progress = last_log = time.monotonic()
    content, reasoning, refusal, calls = [], [], [], {}
    finish = usage = request_id = first_token = None
    stream = None

    def update():
        metrics.update(elapsed_seconds=round(time.monotonic() - started, 3),
                       content_chars=sum(map(len, content)),
                       reasoning_chars=sum(map(len, reasoning)),
                       function_argument_chars=sum(len(c['arguments']) for c in calls.values()),
                       first_token_seconds=first_token, finish_reason=finish)

    def remaining():
        now = time.monotonic()
        total_left, idle_left = total_timeout - (now - started), idle_timeout - (now - last_progress)
        if min(total_left, idle_left) <= 0:
            reason = 'total_deadline' if total_left <= 0 else 'idle_timeout'
            metrics['interruption'] = reason
            raise LLMStreamInterruptedError(f'LLM stream reached {reason}; partial output was discarded')
        return min(total_left, idle_left)

    try:
        options = dict(arguments, stream=True, stream_options={'include_usage': True})
        stream = await asyncio.wait_for(client.chat.completions.create(**options), remaining())
        response = getattr(stream, 'response', None)
        request_id = response.headers.get('x-request-id') if response is not None else None
        iterator = stream.__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), remaining())
            except StopAsyncIteration:
                break
            now = time.monotonic()
            if getattr(chunk, 'usage', None) is not None:
                usage = chunk.usage
            progressed = False
            for choice in chunk.choices or []:
                if choice.index != 0:
                    raise LLMStreamInterruptedError('Only one streamed completion is supported')
                delta = choice.delta
                piece = getattr(delta, 'content', None) or ''
                thought = (getattr(delta, 'reasoning', None) or getattr(delta, 'reasoning_content', None) or '')
                denied = getattr(delta, 'refusal', None) or ''
                content.append(piece)
                reasoning.append(thought)
                refusal.append(denied)
                progressed = progressed or bool(piece or thought or denied)
                for fragment in getattr(delta, 'tool_calls', None) or []:
                    call = calls.setdefault(fragment.index, {'id': '', 'type': 'function', 'name': '', 'arguments': ''})
                    if fragment.id:
                        call['id'] = fragment.id
                    if fragment.type:
                        call['type'] = fragment.type
                    function = fragment.function
                    if function is not None:
                        call['name'] += function.name or ''
                        call['arguments'] += function.arguments or ''
                        progressed = progressed or bool(function.name or function.arguments)
                finish = choice.finish_reason or finish
            if progressed:
                last_progress = now
                if first_token is None:
                    first_token = round(now - started, 3)
            update()
            if now - last_log >= 30:
                logger.info('LLM generation active: %.1fs, %d content / %d function-argument characters',
                            now - started, metrics['content_chars'], metrics['function_argument_chars'])
                last_log = now
            if finish is not None:


                break
        if finish is None:
            metrics['interruption'] = 'missing_finish_reason'
            raise LLMStreamInterruptedError('LLM stream ended without a finish reason; partial output was discarded')
        message = SimpleNamespace(
            content=''.join(content) or None, refusal=''.join(refusal) or None,
            reasoning=''.join(reasoning),
            tool_calls=[SimpleNamespace(id=c['id'], type=c['type'], function=SimpleNamespace(
                name=c['name'], arguments=c['arguments'])) for _, c in sorted(calls.items())] or None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                               usage=usage, _request_id=request_id)
    except asyncio.TimeoutError as exc:
        now = time.monotonic()
        metrics['interruption'] = 'total_deadline' if now - started >= total_timeout else 'idle_timeout'
        raise LLMStreamInterruptedError(f"LLM stream reached {metrics['interruption']}; no duplicate request was submitted") from exc
    except APIConnectionError as exc:
        if first_token is not None:
            metrics['interruption'] = 'connection_lost_after_output'
            raise LLMStreamInterruptedError('LLM connection lost after partial output; no duplicate request was submitted') from exc
        raise
    except (httpx.TimeoutException, httpx.ReadError) as exc:
        metrics['interruption'] = 'stream_transport_error'
        raise LLMStreamInterruptedError('LLM stream stalled or disconnected; no duplicate request was submitted') from exc
    finally:
        update()
        if stream is not None:
            await stream.close()
