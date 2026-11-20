from __future__ import annotations

import asyncio
import base64
import copy
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import logging
import math
import os
import random
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from jsonschema import Draft202012Validator
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, OpenAI

from .llm_stream import LLMStreamInterruptedError, collect_completion

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


class LLMProtocolError(ValueError):
    pass

class LLMRefusalError(RuntimeError):
    pass

class LLMOutputTruncatedError(LLMProtocolError):
    pass

def parse_json(content: str) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise LLMProtocolError(f"Duplicate JSON field: {key}")
            value[key] = item
        return value

    def invalid_constant(value):
        raise LLMProtocolError(f"Nonfinite JSON number: {value}")

    try:
        return json.loads(content, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LLMProtocolError(f"Invalid JSON response: {exc}") from exc


def validate_json(value: Any, schema: dict) -> Any:
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise LLMProtocolError("JSON numbers must be finite")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    Draft202012Validator.check_schema(schema)
    error = next(Draft202012Validator(schema).iter_errors(value), None)
    if error is not None:
        path = ".".join(map(str, error.absolute_path)) or "<root>"
        raise LLMProtocolError(f"Response schema violation at {path}: {error.message}")
    return value


def strict_schema(schema: dict) -> dict:
    Draft202012Validator.check_schema(schema)
    if schema.get("type") != "object":
        raise ValueError("A structured response/function schema must have an object root")

    def visit(node):
        if not isinstance(node, dict):
            return
        kind = node.get("type")
        if kind == "object" or isinstance(kind, list) and "object" in kind:
            properties = node.get("properties", {})
            if node.get("additionalProperties") is not False:
                raise ValueError("Strict schemas require additionalProperties: false for every object")
            if set(node.get("required", [])) != set(properties):
                raise ValueError("Strict schemas must require every object property (use null for optional values)")
        for key in ("properties", "$defs", "definitions"):
            for child in node.get(key, {}).values():
                visit(child)
        for child in node.get("anyOf", []):
            visit(child)
        for child in node.get("oneOf", []):
            visit(child)
        if isinstance(node.get("items"), dict):
            visit(node["items"])

    visit(schema)
    return copy.deepcopy(schema)


def schema_for_api(schema: dict, *, omit_nonblank_pattern: bool = False) -> dict:
    result = copy.deepcopy(schema)
    if omit_nonblank_pattern:
        def visit(node):
            if isinstance(node, dict):
                if node.get('type') == 'string' and node.get('pattern') in {
                        r'\S', r'[\s\S]*\S[\s\S]*'}:
                    node.pop('pattern')
                for child in node.values():
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)
        visit(result)
    return result


def load_settings(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path or ROOT / "config/llm.yaml").expanduser().resolve()
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))["llm"]
    secret_file = settings.get("secrets_file", "config/secrets.env")
    secret_path = Path(secret_file) if secret_file else None
    if secret_path is not None and not secret_path.is_absolute():
        secret_path = ROOT / secret_path
    if secret_path is not None and secret_path.is_file():
        for line in secret_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))
    for env, key in [("NGAA_BASE_URL", "base_url"), ("NGAA_MODEL", "model"),
                     ("AGENT4AIGC_LLM_PROXY", "proxy")]:
        if env in os.environ:
            settings[key] = os.environ[env]
    return settings


def image_url(path: str | Path, *, pixel_budget: int = 2_500_000) -> str:
    value = str(path)
    if value.startswith(("http://", "https://", "data:image/")):
        return value
    from PIL import Image, ImageOps
    local = Path(value).expanduser()
    with Image.open(local) as source:
        im = ImageOps.exif_transpose(source)
        max_edge = int(os.getenv("AGENT4AIGC_LLM_MAX_IMAGE_EDGE", "2048"))
        scale = min(1.0, (pixel_budget / (im.width * im.height)) ** 0.5,
                    max_edge / max(im.size))
        if scale < 1:
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                           Image.Resampling.LANCZOS)
        output = BytesIO()
        if im.mode in {"RGBA", "LA"} or (im.mode == "P" and "transparency" in im.info):
            im.save(output, format="PNG")
            mime = "image/png"
        else:
            im.convert("RGB").save(output, format="JPEG", quality=90, optimize=True)
            mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(output.getvalue()).decode("ascii")


def normalize_content(content: Any) -> Any:
    if content is None or isinstance(content, str):
        return content
    image_count = sum(part.get("type") in {"image", "image_url"} for part in content)
    pixel_budget = max(262_144, 2_500_000 // max(1, image_count))
    parts = []
    for part in content:
        if part.get("type") == "image":
            parts.append({"type": "image_url", "image_url": {"url": image_url(part["image"], pixel_budget=pixel_budget)}})
        elif part.get("type") == "image_url":
            value = part["image_url"]
            if isinstance(value, str):
                value = {"url": value}
            parts.append({"type": "image_url", "image_url": {**value, "url": image_url(value["url"], pixel_budget=pixel_budget)}})
        elif part.get("type") == "text":
            value = part.get("text", "")
            if isinstance(value, list):
                value = "\n".join(map(str, value))
            parts.append({"type": "text", "text": str(value)})
        else:
            raise ValueError(f"Unsupported message content type: {part.get('type')}")
    return parts


class MLLM:
    def __init__(self, model_name: str | None = None, snapshot: bool = False,
                 max_completion_tokens: int | None = None,
                 config_path: str | Path | None = None):
        self.settings = load_settings(config_path)
        self.model = model_name or self.settings["model"]
        self.max_completion_tokens = max_completion_tokens or int(self.settings.get("max_tokens", 4096))
        key_env = self.settings.get("api_key_env", "NGAA_API_KEY")
        key = os.environ.get(key_env) or self.settings.get("api_key")
        if not key:
            raise RuntimeError(f"Set {key_env} or configure api_key in config/llm.yaml")
        proxy = self.settings.get("proxy") or None
        self._http = httpx.Client(proxy=proxy, trust_env=False,
                                  timeout=httpx.Timeout(float(self.settings.get("timeout", 180)),
                                                        connect=15.0, write=60.0, pool=15.0))
        self.client = OpenAI(api_key=key, base_url=self.settings["base_url"],
                             http_client=self._http,
                             max_retries=0)

    @staticmethod
    def _transient(error):
        if isinstance(error, APIConnectionError):
            return True
        if isinstance(error, APIStatusError):
            if error.response.headers.get("x-should-retry", "").lower() == "false":
                return False
            return error.status_code in {408, 409, 429} or error.status_code >= 500
        return False

    @staticmethod
    def _retry_after(error):
        if not isinstance(error, APIStatusError):
            return None
        headers = error.response.headers
        for header, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
            value = headers.get(header)
            if value is None:
                continue
            try:
                delay = float(value) / divisor
            except (ValueError, TypeError):
                if header != "retry-after":
                    continue
                try:
                    date = parsedate_to_datetime(value)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    delay = date.timestamp() - time.time()
                except (ValueError, TypeError, OverflowError):
                    continue
            if math.isfinite(delay) and delay >= 0:
                return delay
        return None

    def _request_with_retries(self, arguments, events):
        settings = self.settings
        for flag in ('stream', 'retry_timeouts'):
            if flag in settings and type(settings[flag]) is not bool:
                raise ValueError(f'llm.{flag} must be a boolean')
        retries = settings.get("max_retries", 0)
        if type(retries) is not int or not 0 <= retries <= 10:
            raise ValueError("llm.max_retries must be an integer from 0 to 10")
        base = float(settings.get("retry_backoff_base", 2))
        cap = float(settings.get("retry_max_backoff", 10))
        jitter = float(settings.get("retry_jitter", 0.1))
        budget = float(settings.get("retry_budget", 600))
        if any(not math.isfinite(x) for x in (base, cap, jitter, budget)) or min(base, cap, jitter) < 0 or budget <= 0:
            raise ValueError("Retry delays must be finite/nonnegative and retry_budget positive")
        started = time.monotonic()
        for attempt in range(1, retries + 2):
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("LLM transport retry budget exhausted")
            options = dict(arguments)
            requested_timeout = options.get("timeout", getattr(self.client, "timeout", settings.get("timeout", 180)))
            if isinstance(requested_timeout, httpx.Timeout):
                values = requested_timeout.as_dict()
                options["timeout"] = httpx.Timeout(**{
                    key: min(value, remaining) if value is not None else remaining
                    for key, value in values.items()})
            else:
                options["timeout"] = min(float(requested_timeout), remaining)
            event = {"attempt": attempt, "retry_budget_remaining_seconds": round(remaining, 3)}
            streaming = settings.get('stream', False)
            total_timeout = None
            if streaming:
                idle = float(settings.get('stream_idle_timeout', 90))
                limit = float(settings.get('stream_max_seconds', 1200))
                if not math.isfinite(idle) or not math.isfinite(limit) or min(idle, limit) <= 0:
                    raise ValueError('Stream idle and total limits must be finite and positive')
                configured = (options['timeout'].read if isinstance(options['timeout'], httpx.Timeout)
                              else options['timeout'])

                total_timeout = min(remaining, limit, max(configured or 0, 120 + arguments.get('max_tokens', 0) / 4))
                if isinstance(options['timeout'], httpx.Timeout):
                    values = options['timeout'].as_dict()
                    values['read'] = min(idle, remaining)
                    options['timeout'] = httpx.Timeout(**values)
                else:
                    options['timeout'] = httpx.Timeout(min(idle, remaining))
                event['stream'] = {'idle_timeout_seconds': min(idle, remaining),
                                   'total_timeout_seconds': round(total_timeout, 3)}
            if isinstance(options['timeout'], httpx.Timeout):
                event['http_phase_timeouts'] = options['timeout'].as_dict()
                event['timeout_seconds'] = max(event['http_phase_timeouts'].values())
            else:
                event['timeout_seconds'] = options['timeout']
            attempt_started = time.monotonic()
            try:
                if streaming:
                    async def receive():
                        async with AsyncOpenAI(api_key=self.client.api_key, base_url=self.client.base_url,
                                               max_retries=0, http_client=httpx.AsyncClient(
                                                   proxy=settings.get('proxy') or None, trust_env=False)) as client:
                            return await collect_completion(client, options, idle_timeout=min(idle, remaining),
                                                            total_timeout=total_timeout, metrics=event['stream'])
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        response = asyncio.run(receive())
                    else:

                        with ThreadPoolExecutor(max_workers=1) as pool:
                            response = pool.submit(lambda: asyncio.run(receive())).result()
                else:
                    response = self.client.chat.completions.create(**options)
            except Exception as exc:
                elapsed = time.monotonic() - attempt_started
                event.update(status="failed", elapsed_seconds=round(elapsed, 3),
                             error_type=type(exc).__name__)
                if isinstance(exc, APIStatusError):
                    event.update(http_status=exc.status_code, request_id=exc.request_id)
                transient = self._transient(exc)
                if isinstance(exc, LLMStreamInterruptedError):
                    transient = False
                    event['retry_skipped'] = 'incomplete_stream_not_replayed'
                elif isinstance(exc, APITimeoutError) and not settings.get('retry_timeouts', True):
                    transient = False
                    event['retry_skipped'] = 'timeout_retry_disabled'
                event["retryable"] = transient
                remaining = budget - (time.monotonic() - started)
                delay = self._retry_after(exc)
                if delay is None:
                    delay = min(cap, base * 2 ** (attempt - 1))
                    delay *= 1 + random.uniform(0, jitter)
                    delay = min(cap, delay)
                retry = transient and attempt <= retries and delay < remaining
                event["will_retry"] = retry
                if retry:
                    event["retry_delay_seconds"] = round(delay, 3)
                elif transient and attempt <= retries:
                    event["retry_skipped"] = "budget_exhausted"
                if events is not None:
                    events.append(event)
                if not retry:
                    raise
                logger.warning("LLM %s on attempt %d/%d; retrying in %.1fs (budget left %.1fs)",
                               type(exc).__name__, attempt, retries + 1, delay, remaining)
                time.sleep(delay)
            else:
                event.update(status="passed", elapsed_seconds=round(time.monotonic() - attempt_started, 3),
                             request_id=getattr(response, "_request_id", None))
                if events is not None:
                    events.append(event)
                return response

    def _completion(self, messages: list[dict], **kwargs: Any):
        transport_events = kwargs.pop("_transport_events", None)
        extra_body = copy.deepcopy(self.settings.get("extra_body") or {})
        extra_body.update(kwargs.pop("extra_body", None) or {})
        if extra_body:
            kwargs["extra_body"] = extra_body
        normalized = [{**msg, "content": normalize_content(msg["content"])} for msg in messages]
        arguments = dict(model=self.model, messages=normalized,
                         max_tokens=kwargs.pop("max_tokens", self.max_completion_tokens),
                         temperature=kwargs.pop("temperature", self.settings.get("temperature", 0.3)), **kwargs)
        response = self._request_with_retries(arguments, transport_events)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            error = LLMOutputTruncatedError("LLM response was truncated; return concise complete fields within the token budget")
            calls = getattr(choice.message, "tool_calls", None)
            error.raw = ([{"name": call.function.name, "arguments": call.function.arguments}
                          for call in calls] if calls else choice.message.content)
            raise error
        if choice.finish_reason == "content_filter":
            raise LLMRefusalError("LLM response was blocked by the provider's content filter")
        if getattr(choice.message, "refusal", None):
            raise LLMRefusalError(f"LLM refused the request: {choice.message.refusal}")
        return choice.message

    def generate(self, messages: list[dict], **kwargs: Any) -> str:
        content = self._completion(messages, **kwargs).content
        if not content:
            raise RuntimeError("LLM returned empty content")
        return content

    @staticmethod
    def _messages(prompt: Any, history: list[dict] | None,
                  system_msgs: list[str] | None) -> list[dict]:
        messages = [{"role": "system", "content": item} for item in (system_msgs or [])]
        messages += copy.deepcopy(history or [])
        messages.append({"role": "user", "content": prompt})
        return messages

    def _structured_attempts(self, messages, options, parse, *, max_attempts, protocol_events):
        if type(max_attempts) is not int or max_attempts not in (1, 2):
            raise ValueError("max_attempts must be 1 or 2")
        original = copy.deepcopy(messages)
        for attempt in range(1, max_attempts + 1):
            event = {"attempt": attempt, "transport_attempts": []}
            raw = None
            try:
                message = self._completion(messages, _transport_events=event["transport_attempts"], **options)
                calls = getattr(message, "tool_calls", None)
                raw = ([{"name": call.function.name, "arguments": call.function.arguments}
                        for call in calls] if calls else message.content)
                event["raw"] = raw
                result = parse(message)
            except LLMProtocolError as exc:
                if raw is None and getattr(exc, "raw", None) is not None:
                    raw = exc.raw
                    event["raw"] = raw
                event.update(status="protocol_error", error_type=type(exc).__name__, error=str(exc))
                if protocol_events is not None:
                    protocol_events.append(event)
                if attempt == max_attempts:
                    raise LLMProtocolError(f"Protocol validation failed after {attempt} attempt(s): {exc}") from exc
                repair = ("The previous response violated the required contract: " + str(exc) +
                          ". Repair the serialization/fields, preserving the original task and evidence. "
                          "Use concise complete values and the same required schema or function.")
                if raw is not None:
                    repair += "\nRejected response: " + (raw if isinstance(raw, str) else
                                                        json.dumps(raw, ensure_ascii=False, allow_nan=False))
                messages = copy.deepcopy(original) + [{"role": "user", "content": repair}]
            except Exception as exc:
                event.update(status="request_error", error_type=type(exc).__name__, error=str(exc))
                if protocol_events is not None:
                    protocol_events.append(event)
                raise
            else:
                event.update(status="passed", output=result)
                if protocol_events is not None:
                    protocol_events.append(event)
                return result

    def request_json(self, prompt: Any, *, schema: dict, name: str = "agent_message",
                     history: list[dict] | None = None,
                     system_msgs: list[str] | None = None, max_attempts: int = 2,
                     protocol_events: list | None = None, validate=None, **kwargs: Any) -> dict:
        contract = strict_schema(schema)
        def parse(message):
            if not message.content:
                raise LLMProtocolError("Structured output returned no JSON content")
            result = validate_json(parse_json(message.content), contract)
            if validate is not None:
                try:
                    validate(result)
                except ValueError as exc:
                    raise LLMProtocolError(str(exc)) from exc
            return result
        wire_schema = schema_for_api(contract, omit_nonblank_pattern=self.settings.get('omit_nonblank_pattern', False))
        options = {**kwargs, "response_format": {"type": "json_schema", "json_schema": {
            "name": name, "strict": True, "schema": wire_schema}}}
        return self._structured_attempts(self._messages(prompt, history, system_msgs), options, parse,
                                         max_attempts=max_attempts, protocol_events=protocol_events)

    async def arequest_json(self, prompt: Any, **kwargs: Any) -> dict:
        return await asyncio.to_thread(self.request_json, prompt, **kwargs)

    def request_tool(self, prompt: Any, *, tools: list[dict], tool_choice: Any = "required",
                     history: list[dict] | None = None,
                     system_msgs: list[str] | None = None, max_attempts: int = 2,
                     protocol_events: list | None = None, validate=None, **kwargs: Any) -> dict:
        specifications = copy.deepcopy(tools)
        offered = {}
        for spec in specifications:
            if spec.get("type") != "function":
                raise ValueError("Only official function tool specifications are supported")
            function = spec["function"]
            name = function["name"]
            if name in offered:
                raise ValueError(f"Duplicate tool specification: {name}")
            function["parameters"] = strict_schema(function["parameters"])
            function["strict"] = True
            offered[name] = function["parameters"]
            function["parameters"] = schema_for_api(
                function["parameters"], omit_nonblank_pattern=self.settings.get('omit_nonblank_pattern', False))
        if not offered:
            raise ValueError("At least one tool must be offered")
        if isinstance(tool_choice, dict):
            forced = tool_choice.get("function", {}).get("name")
            if tool_choice.get("type") != "function" or forced not in offered:
                raise ValueError("tool_choice must refer to an offered function")
        elif tool_choice != "required":
            raise ValueError("request_tool requires a forced function or tool_choice='required'")
        def parse(message):
            calls = message.tool_calls or []
            if len(calls) != 1:
                raise LLMProtocolError(f"Expected exactly one tool call, received {len(calls)}")
            call = calls[0]
            name = call.function.name
            if call.type != "function" or name not in offered:
                raise LLMProtocolError(f"Model selected an unoffered function: {name}")
            if isinstance(tool_choice, dict) and name != tool_choice["function"]["name"]:
                raise LLMProtocolError(f"Model ignored forced function selection: {name}")
            arguments = validate_json(parse_json(call.function.arguments), offered[name])
            result = {"name": name, "arguments": arguments}
            if validate is not None:
                try:
                    validate(result)
                except ValueError as exc:
                    raise LLMProtocolError(str(exc)) from exc
            return result
        options = {**kwargs, "tools": specifications, "tool_choice": tool_choice, "parallel_tool_calls": False}
        return self._structured_attempts(self._messages(prompt, history, system_msgs), options, parse,
                                         max_attempts=max_attempts, protocol_events=protocol_events)

    async def arequest_tool(self, prompt: Any, **kwargs: Any) -> dict:
        return await asyncio.to_thread(self.request_tool, prompt, **kwargs)

    async def agenerate(self, messages: list[dict], **kwargs: Any) -> str:
        return await asyncio.to_thread(self.generate, messages, **kwargs)

    def chat(self, prompt: Any, history: list[dict] | None = None) -> tuple[str, list[dict]]:
        messages = copy.deepcopy(history or [])
        messages.append({"role": "user", "content": prompt})
        reply = self.generate(messages)
        messages.append({"role": "assistant", "content": reply})
        return reply, messages

    async def achat(self, prompt: Any, history: list[dict] | None = None) -> tuple[str, list[dict]]:
        return await asyncio.to_thread(self.chat, prompt, history)

    async def aask(self, prompt: Any, system_msgs: list[str] | None = None,
                   format_msgs: list[dict] | None = None, **kwargs: Any) -> str:
        messages = [{"role": "system", "content": item} for item in (system_msgs or [])]
        messages += copy.deepcopy(format_msgs or [])
        messages.append({"role": "user", "content": prompt})
        return await self.agenerate(messages, **kwargs)

    def llm_response(self, user_input: str, images: list[str] | None = None,
                     sys_input: str | None = None) -> str:
        messages = [{"role": "system", "content": sys_input}] if sys_input else []
        content = [{"type": "text", "text": user_input}]
        content += [{"type": "image", "image": p} for p in (images or [])]
        messages.append({"role": "user", "content": content})
        return self.generate(messages)

    def close(self) -> None:
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
