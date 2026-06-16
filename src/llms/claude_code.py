"""Anthropic Claude Code CLI (``claude``) provider — headless structured output.

``claude --print --output-format json --json-schema <schema> --tools ""`` runs the
CLI as a pure single-shot reasoning model (all tools disabled) and returns a JSON
envelope whose ``structured_output`` field is the schema-validated object. This
adapter flattens the structured-messages format to one prompt, drives the CLI, and
parses ``structured_output`` (falling back to the free-text ``result``) into the
requested pydantic model.

Auth is whatever the local ``claude`` CLI is already logged into (a Claude
Pro/Max subscription via ``~/.claude/.credentials.json``, or the
``CLAUDE_CODE_OAUTH_TOKEN`` env var from ``claude setup-token``), so no API key is
billed. Each call runs in its own temp cwd with session persistence disabled, so
parallel calls don't collide and no project ``CLAUDE.md`` leaks into the prompt.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import tempfile
import time
import typing as T
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from src.llms.models import TokenUsage
from src.log import log

BMType = T.TypeVar("BMType", bound=BaseModel)

_SESSION_TIMEOUT_SECONDS = 3600
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Claude Code refuses to run as root without this; harmless otherwise.
os.environ.setdefault("IS_SANDBOX", "1")

# --- Session-usage limiter ---------------------------------------------------
# The Claude subscription "session" limit is the rolling 5-hour usage window,
# read from the OAuth usage endpoint. Before each call we pause until the window
# resets if usage has reached the threshold (the agy quota-cooldown pattern), so
# the run keeps its place and continues once capacity frees up.
_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
_OAUTH_BETA = "oauth-2025-04-20"
# Wait once the 5-hour session window is this full (0..1). 0 disables.
_SESSION_LIMIT_THRESHOLD = float(os.environ.get("CLAUDE_CODE_SESSION_LIMIT_THRESHOLD", "0.70"))
_USAGE_PROBE_TIMEOUT_S = 30
_RESET_HEADROOM_S = 120
_MAX_WAIT_S = 6 * 3600
_MAX_WAIT_CYCLES = 4
_USAGE_CACHE_TTL_S = 30  # share one probe result across concurrent calls

_usage_lock = asyncio.Lock()
_usage_cache: tuple[float, tuple[float, float] | None] | None = None  # (checked_monotonic, result)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _oauth_token() -> str | None:
    """The subscription OAuth bearer: env token, else the CLI's on-disk creds."""
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if token:
        return token
    try:
        data = json.loads((Path.home() / ".claude" / ".credentials.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    oauth = data.get("claudeAiOauth")
    return oauth.get("accessToken") if isinstance(oauth, dict) else None


def _seconds_until(iso_ts: str | None) -> float:
    if not iso_ts:
        return float(_MAX_WAIT_S)
    try:
        reset = datetime.fromisoformat(iso_ts)
    except ValueError:
        return float(_MAX_WAIT_S)
    if reset.tzinfo is None:
        reset = reset.replace(tzinfo=UTC)
    return max(0.0, (reset - datetime.now(UTC)).total_seconds())


def _fetch_session_usage_blocking() -> tuple[float, float] | None:
    """Return ``(utilization_percent, seconds_until_reset)`` for the 5-hour
    session window, or ``None`` if it can't be read (callers fail open)."""
    token = _oauth_token()
    if not token:
        return None
    req = urllib.request.Request(
        _USAGE_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-beta": _OAUTH_BETA,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_USAGE_PROBE_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    five_hour = payload.get("five_hour")
    if not isinstance(five_hour, dict) or five_hour.get("utilization") is None:
        return None
    return float(five_hour["utilization"]), _seconds_until(five_hour.get("resets_at"))


async def _probe_session_usage() -> tuple[float, float] | None:
    """Cached probe shared across concurrent calls so they don't all hammer the
    endpoint; the off-thread GET runs at most once per TTL window."""
    global _usage_cache
    async with _usage_lock:
        if _usage_cache is not None and time.monotonic() - _usage_cache[0] < _USAGE_CACHE_TTL_S:
            return _usage_cache[1]
        result = await asyncio.to_thread(_fetch_session_usage_blocking)
        _usage_cache = (time.monotonic(), result)
        return result


async def _wait_for_session_capacity() -> None:
    """Block until the 5-hour session window has room, then return. Probe
    failures fail open (log and proceed) rather than stall the whole run."""
    if _SESSION_LIMIT_THRESHOLD <= 0:
        return
    threshold_pct = _SESSION_LIMIT_THRESHOLD * 100
    for _ in range(_MAX_WAIT_CYCLES):
        usage = await _probe_session_usage()
        if usage is None:
            log.warn("claudecode_usage_check_failed", note="proceeding without session limit")
            return
        utilization, reset_s = usage
        if utilization < threshold_pct:
            return
        sleep_s = min(reset_s + _RESET_HEADROOM_S, _MAX_WAIT_S)
        log.warn(
            "claudecode_session_limit_wait",
            utilization=utilization,
            threshold=threshold_pct,
            sleep_minutes=round(sleep_s / 60, 1),
        )
        global _usage_cache
        _usage_cache = None  # force a fresh probe after the window resets
        await asyncio.sleep(sleep_s)
    log.warn("claudecode_session_limit_still_high", note="proceeding after max wait cycles")


def _flatten_messages(messages: list[dict]) -> str:
    """Collapse the structured-messages format to a single prompt string.

    Mirrors :func:`src.llms.structured.update_messages_gemini`; images are
    dropped (the headless ``--print`` surface is text-only here).
    """
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        if isinstance(content, list):
            text_parts = [
                c.get("text", c.get("content", ""))
                for c in content
                if c.get("type") in {"input_text", "output_text", "text"}
            ]
            text = "\n".join(p for p in text_parts if p)
        else:
            text = str(content or "")
        if not text:
            continue
        prefix = {"system": "System", "user": "User", "assistant": "Assistant"}.get(
            role, role.capitalize()
        )
        parts.append(f"{prefix}: {text}")
    return "\n\n".join(parts)


def _usage_from_envelope(usage: dict | None) -> TokenUsage:
    if not isinstance(usage, dict):
        return TokenUsage()
    input_tokens = int(usage.get("input_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return TokenUsage(
        input_tokens=input_tokens + cache_creation,
        output_tokens=output_tokens,
        cached_tokens=cache_read,
        total_tokens=input_tokens + cache_creation + cache_read + output_tokens,
    )


async def _get_next_structure_claudecode(
    structure: type[BMType],
    model_id: str,
    messages: list,
) -> tuple[BMType, TokenUsage]:
    base_path = shutil.which("claude")
    if not base_path:
        raise RuntimeError("`claude` executable not found in PATH")

    await _wait_for_session_capacity()

    schema = structure.model_json_schema()
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    prompt = _flatten_messages(messages)

    with tempfile.TemporaryDirectory(prefix="claude-cc-") as ws_str:
        cmd = [
            base_path,
            "--print",
            "--output-format",
            "json",
            "--model",
            model_id,
            "--tools",
            "",
            "--no-session-persistence",
            "--setting-sources",
            "",
            "--json-schema",
            json.dumps(schema),
            prompt,
        ]
        # Force the CLI's subscription OAuth login (~/.claude/.credentials.json or
        # CLAUDE_CODE_OAUTH_TOKEN). The project's ANTHROPIC_API_KEY would otherwise
        # take precedence and route to the paid API — the opposite of why we drive
        # the CLI — so strip it from the child env only.
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=ws_str,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=_SESSION_TIMEOUT_SECONDS
            )
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            stdout_b, stderr_b = await proc.communicate()

    stdout = _strip_ansi((stdout_b or b"").decode("utf-8", errors="replace")).strip()
    stderr = _strip_ansi((stderr_b or b"").decode("utf-8", errors="replace")).strip()

    if not stdout:
        raise RuntimeError(f"claude returned no stdout (stderr={stderr[:300]!r})")

    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"claude output was not JSON: {exc}; stdout head={stdout[:300]!r}"
        ) from exc

    if envelope.get("is_error"):
        raise RuntimeError(
            f"claude reported an error: {envelope.get('result') or envelope.get('subtype')!r}"
        )

    usage = _usage_from_envelope(envelope.get("usage"))

    structured = envelope.get("structured_output")
    if structured is not None:
        return structure.model_validate(structured), usage

    # Fall back to parsing the free-text result if structured output is absent
    # (e.g. the model wrapped the JSON in prose despite the schema).
    result_text = envelope.get("result")
    if isinstance(result_text, str) and result_text.strip():
        start, end = result_text.find("{"), result_text.rfind("}")
        if start != -1 and end > start:
            with contextlib.suppress(Exception):
                return structure.model_validate_json(result_text[start : end + 1]), usage

    log.warn("claudecode_no_structured_output", envelope_keys=list(envelope.keys()))
    raise RuntimeError(
        f"claude returned no structured_output for {structure.__name__}; "
        f"result head={str(envelope.get('result'))[:300]!r}"
    )
