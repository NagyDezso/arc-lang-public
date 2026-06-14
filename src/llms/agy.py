"""Google Antigravity CLI (``agy``) provider — headless one-shot mode.

``agy --print`` only emits the model's final plain-text answer; there is no
``--model`` flag and no structured-output mode. This adapter:

1. Pins the model via ``~/.gemini/antigravity-cli/settings.json``.
2. Pipes a statusline shell script onto the same settings so each render appends
   the JSON token-usage payload to ``usage.jsonl`` — the only way to surface
   billed tokens in ``--print`` mode.
3. Clamps every prompt with an explicit "do NOT call any tools" instruction
   (the only knob that reliably blocks agy's builtin tools — ``disabledTools``
   in settings does not gate them).
4. Parses ``agy``'s reply as JSON matching the requested pydantic schema.
5. Retries on quota-exhaustion (``agy`` exits silently on RESOURCE_EXHAUSTED;
   the reset window is parsed from its log file).
6. Saves a copy of each conversation transcript under ``AGY_TRANSCRIPT_DIR`` so
   :func:`src.submit.check_transcripts` can scan them for tool use / network
   access after the run.

Concurrency: each call runs in its own temp ``$HOME`` so the per-render token
payloads, transcript brain dir, and orchestrator log are fully isolated.
``~/.cache`` (Playwright, ~128MB) and the read-only bits of
``~/.gemini/antigravity-cli/`` (binaries, onboarding cache, builtin skills) are
symlinked from the real user home so each call only pays for a handful of
small files of setup. No lock — calls run truly in parallel.
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
import uuid
from pathlib import Path

from pydantic import BaseModel

from src.llms.models import TokenUsage
from src.log import log
from src.logging_config import get_task_id

BMType = T.TypeVar("BMType", bound=BaseModel)


_DATA_DIR = Path(
    os.environ.get("AGY_DATA_DIR") or (Path.home() / ".gemini" / "antigravity-cli")
)
_USAGE_LOG_NAME = "usage.jsonl"
_STATUSLINE_SCRIPT_NAME = "statusline-usage.sh"

# agy's settings.json `model` field expects a human-readable display name, not
# the API id.
_MODEL_DISPLAY_NAMES = {
    "gemini-3.5-flash": "Gemini 3.5 Flash (Medium)",
    "gemini-3.5-flash-high": "Gemini 3.5 Flash (High)",
    "claude-sonnet-4-6": "Claude Sonnet 4.6 (Thinking)",
}

# Passed to `agy --print-timeout` (Go duration); keep it >= the per-call wall-
# clock cap.
_PRINT_TIMEOUT = "60m"
_SESSION_TIMEOUT_SECONDS = 3600

# `agy --print` exits silently on quota exhaustion; any run finishing this fast
# with no stdout/stderr is treated as a silent failure.
_QUOTA_SILENT_FAILURE_S = 15.0
_QUOTA_COOLDOWN_HEADROOM_S = 120
_QUOTA_FALLBACK_RESET_S = 5 * 3600
_MAX_QUOTA_RETRIES = 3
_MAX_PARSE_RETRIES = 2
_QUOTA_RESET_RE = re.compile(
    r"Resets in\s+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?",
    re.IGNORECASE,
)

_NO_TOOLS_CLAMP = """\
CRITICAL CONSTRAINT: You MUST NOT call any tools. Do not search the web. Do not
read, write, edit, or list any files. Do not run any commands. Do not open a
browser. Do not schedule anything. Solve the task entirely in your own reasoning
and reply with plain text only.
"""

_JSON_INSTRUCTION_TEMPLATE = """\
============================================================
RESPONSE FORMAT — READ CAREFULLY
============================================================
Your reply MUST be a single JSON object that is an INSTANCE matching the schema
below. The schema describes the SHAPE your response must take; it is NOT the
response itself.

You MUST NOT:
- Echo the schema back (do not output the keys `properties`, `required`,
  `type`, `items`, `additionalProperties`, `$defs`, `$schema` at the top level)
- Wrap the JSON in markdown code fences
- Write any prose, thinking, "wait", "hmm", or commentary before or after
- Output anything other than the JSON object

Your reply MUST:
- Start with `{{` and end with `}}`
- Contain every key listed under `required` in the schema
- Be valid parseable JSON (every array closed, every string quoted, no
  trailing text)

JSON Schema (this is the SHAPE; produce an instance of it):
{schema}
"""


_PER_CALL_DATA_NAMES = frozenset({
    "settings.json",
    "statusline-usage.sh",
    "usage.jsonl",
    "brain",
    "log",
})


def _init_home_for_call(home: Path) -> Path:
    """Build a fresh temp ``$HOME`` that agy can run in without colliding with
    concurrent calls.

    Large read-only resources (``~/.cache`` for Playwright, ``bin/`` for agy
    binaries, ``cache/`` for onboarding) are symlinked from the real user home.
    Per-call mutable paths (``settings.json``, ``statusline-usage.sh``,
    ``usage.jsonl``, ``brain/``, ``log/``) are left empty so the caller writes
    them fresh — that's what enables parallel calls to keep separate token
    counts and transcript histories.

    Returns the ``<home>/.gemini/antigravity-cli`` directory.
    """
    real_home = Path.home()

    real_cache = real_home / ".cache"
    if real_cache.exists():
        (home / ".cache").symlink_to(real_cache)

    home_gemini = home / ".gemini"
    home_gemini.mkdir()
    real_gemini = real_home / ".gemini"
    if real_gemini.exists():
        for item in real_gemini.iterdir():
            if item.name == "antigravity-cli":
                continue
            (home_gemini / item.name).symlink_to(item)

    home_agy = home_gemini / "antigravity-cli"
    home_agy.mkdir()
    if _DATA_DIR.exists():
        for item in _DATA_DIR.iterdir():
            if item.name in _PER_CALL_DATA_NAMES:
                continue
            (home_agy / item.name).symlink_to(item)

    return home_agy


def _setup_data_dir(data_dir: Path, model_id: str) -> None:
    """Write the per-call files agy needs (statusline hook + settings)."""
    data_dir.mkdir(parents=True, exist_ok=True)

    refresh_token = os.environ.get("ANTIGRAVITY_OAUTH_REFRESH_TOKEN")
    if refresh_token:
        token_file = data_dir / "antigravity-oauth-token"
        token_file.write_text(
            json.dumps(
                {
                    "token": {
                        "access_token": "",
                        "token_type": "Bearer",
                        "refresh_token": refresh_token,
                        "expiry": "2000-01-01T00:00:00Z",
                    },
                    "auth_method": "consumer",
                }
            )
        )
        token_file.chmod(0o600)

    usage_log = data_dir / _USAGE_LOG_NAME
    statusline_script = data_dir / _STATUSLINE_SCRIPT_NAME
    statusline_script.write_text(
        f'#!/bin/sh\n{{ cat; printf "\\n"; }} >> "{usage_log}"\n'
    )
    statusline_script.chmod(0o755)

    display_name = _MODEL_DISPLAY_NAMES.get(model_id, model_id)
    (data_dir / "settings.json").write_text(
        json.dumps(
            {
                "enableTelemetry": False,
                "model": display_name,
                "statusLine": {
                    "type": "command",
                    "command": str(statusline_script),
                    "enabled": True,
                },
                "toolPermission": "always-proceed",
                "trustedWorkspaces": ["/tmp"],
            },
            indent=2,
        )
    )


def _flatten_messages(messages: list[dict]) -> str:
    """Collapse the structured-messages format to a single prompt string.

    Mirrors what :func:`src.llms.structured.update_messages_gemini` does for
    Gemini direct calls — images are dropped (agy --print has no multimodal
    surface).
    """
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        if isinstance(content, list):
            text_parts: list[str] = []
            for c in content:
                if c.get("type") in {"input_text", "output_text", "text"}:
                    text_parts.append(c.get("text", c.get("content", "")))
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


def _build_prompt(structure: type[BMType], messages: list[dict]) -> str:
    schema = structure.model_json_schema()
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False
    schema.pop("description", None)
    schema.pop("title", None)
    body = _flatten_messages(messages)
    json_instruction = _JSON_INSTRUCTION_TEMPLATE.format(
        schema=json.dumps(schema, indent=2)
    )
    return f"{_NO_TOOLS_CLAMP}\n{body}\n\n{json_instruction}"


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _extract_json_payload(stdout: str) -> str:
    """Pull the first JSON object out of agy's plain-text reply."""
    text = stdout.strip()
    if not text:
        raise ValueError("agy returned empty stdout")
    # Strip Markdown fences if the model added them despite the clamp.
    fenced = _FENCED_JSON_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError(f"no JSON object found in agy stdout: {text[:200]!r}")
    return match.group(0)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def _probe_quota_reset_seconds(log_path: Path) -> tuple[int | None, str]:
    """Parse ``Resets in Xh Ym Zs`` out of agy's RESOURCE_EXHAUSTED log line."""
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None, f"(log file not found: {log_path})"
    except OSError as exc:
        return None, f"(failed to read log {log_path}: {exc})"

    for line in content.splitlines():
        if "RESOURCE_EXHAUSTED" not in line:
            continue
        match = _QUOTA_RESET_RE.search(line)
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds, line.strip()
    return None, content[-4096:]


def _collect_usage(data_dir: Path) -> TokenUsage:
    """Reconstruct billed token usage from statusline payloads in this call's
    isolated data dir.

    agy re-emits ``current_usage`` on every render while a request streams,
    with ``output_tokens`` climbing until the response completes and
    ``input_tokens`` constant. Consecutive payloads sharing the same input are
    therefore one request; its final (max) output is the billed amount.
    """
    usage_log = data_dir / _USAGE_LOG_NAME
    try:
        lines = usage_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return TokenUsage()

    groups: list[list] = []  # [(input, cache_creation, cache_read), max_output]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = (obj.get("context_window") or {}).get("current_usage")
        if not isinstance(usage, dict):
            continue
        key = (
            int(usage.get("input_tokens") or 0),
            int(usage.get("cache_creation_input_tokens") or 0),
            int(usage.get("cache_read_input_tokens") or 0),
        )
        out = int(usage.get("output_tokens") or 0)
        if groups and groups[-1][0] == key:
            groups[-1][1] = max(groups[-1][1], out)
        else:
            groups.append([key, out])

    totals = TokenUsage()
    for (inp, creation, cached), out in groups:
        totals += TokenUsage(
            input_tokens=inp + creation,
            cached_tokens=cached,
            output_tokens=out,
            total_tokens=inp + creation + out,
        )
    return totals


def _collect_new_transcripts(data_dir: Path) -> list[tuple[Path, list[str]]]:
    """Read every transcript line agy produced in this call's brain dir."""
    brain_dir = data_dir / "brain"
    if not brain_dir.is_dir():
        return []
    grown: list[tuple[Path, list[str]]] = []
    for path in sorted(
        brain_dir.glob("*/.system_generated/logs/transcript.jsonl"),
        key=lambda p: p.stat().st_mtime,
    ):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        if lines:
            grown.append((path, lines))
    return grown


def _save_transcripts(new_transcripts: list[tuple[Path, list[str]]]) -> None:
    """Copy new transcript lines to ``AGY_TRANSCRIPT_DIR`` for later auditing."""
    dest_dir_str = os.environ.get("AGY_TRANSCRIPT_DIR")
    if not dest_dir_str or not new_transcripts:
        return
    dest_dir = Path(dest_dir_str)
    dest_dir.mkdir(parents=True, exist_ok=True)
    task_id = get_task_id() or "no-task"
    safe_task = re.sub(r"[^A-Za-z0-9_-]+", "_", task_id) or "no-task"
    for src_path, lines in new_transcripts:
        conv_id = src_path.parent.parent.parent.name  # brain/<uuid>/.../transcript.jsonl
        suffix = uuid.uuid4().hex[:8]
        dest = dest_dir / f"{safe_task}__{conv_id}__{suffix}.jsonl"
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _count_tool_calls(new_transcripts: list[tuple[Path, list[str]]]) -> int:
    count = 0
    for _path, lines in new_transcripts:
        for line in lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tc = obj.get("tool_calls")
            if isinstance(tc, list):
                count += len(tc)
    return count


async def _invoke_agy_once(
    base_path: str,
    prompt: str,
    ws_path: Path,
    log_path: Path,
    home: Path,
) -> tuple[str, str, float]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    cmd = [
        base_path,
        "--dangerously-skip-permissions",
        "--print-timeout",
        _PRINT_TIMEOUT,
        "--log-file",
        str(log_path),
        "--add-dir",
        str(ws_path),
        "--print",
        prompt,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(ws_path),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    start = time.monotonic()
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=_SESSION_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        stdout_b, stderr_b = await proc.communicate()
    elapsed = time.monotonic() - start
    stdout = _strip_ansi((stdout_b or b"").decode("utf-8", errors="replace"))
    stderr = _strip_ansi((stderr_b or b"").decode("utf-8", errors="replace"))
    return stdout, stderr, elapsed


async def _get_next_structure_agy(
    structure: type[BMType],
    model_id: str,
    messages: list,
) -> tuple[BMType, TokenUsage]:
    base_path = shutil.which("agy")
    if not base_path:
        raise RuntimeError("`agy` executable not found in PATH")

    prompt = _build_prompt(structure=structure, messages=messages)

    quota_attempts = 0
    parse_attempts = 0
    last_error: Exception | None = None
    while True:
        with tempfile.TemporaryDirectory(prefix="agy-home-") as home_str:
            home = Path(home_str)
            data_dir = _init_home_for_call(home)
            _setup_data_dir(data_dir, model_id)

            with tempfile.TemporaryDirectory(prefix="agy-ws-") as ws_str:
                ws_path = Path(ws_str)
                log_path = data_dir / "log" / f"orchestrator-{uuid.uuid4().hex}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)

                stdout, stderr, elapsed = await _invoke_agy_once(
                    base_path=base_path,
                    prompt=prompt,
                    ws_path=ws_path,
                    log_path=log_path,
                    home=home,
                )

            new_transcripts = _collect_new_transcripts(data_dir)
            _save_transcripts(new_transcripts)
            tool_call_count = _count_tool_calls(new_transcripts)
            usage = _collect_usage(data_dir)

            silent = (
                not stdout.strip()
                and not stderr.strip()
                and not new_transcripts
                and elapsed < _QUOTA_SILENT_FAILURE_S
            )
            if silent:
                reset_seconds, probe_debug = _probe_quota_reset_seconds(log_path)
                if reset_seconds is None:
                    reset_seconds = _QUOTA_FALLBACK_RESET_S
                    log.warn(
                        "agy_quota_silent_no_reset_parse",
                        elapsed=elapsed,
                        fallback_seconds=reset_seconds,
                        log_tail=probe_debug[-1024:],
                    )
                if quota_attempts >= _MAX_QUOTA_RETRIES:
                    raise RuntimeError(
                        f"agy quota exhausted; gave up after "
                        f"{_MAX_QUOTA_RETRIES + 1} attempts"
                    )
                quota_attempts += 1
                cooldown = reset_seconds + _QUOTA_COOLDOWN_HEADROOM_S
                log.warn(
                    "agy_quota_exhausted",
                    attempt=quota_attempts,
                    sleep_seconds=cooldown,
                )
                await asyncio.sleep(cooldown)
                continue

            if tool_call_count > 0:
                log.warn(
                    "agy_used_tools_despite_clamp",
                    tool_call_count=tool_call_count,
                    elapsed_seconds=elapsed,
                )

            if not stdout.strip():
                last_error = RuntimeError(
                    f"agy returned no stdout (stderr={stderr[:200]!r})"
                )
                if parse_attempts >= _MAX_PARSE_RETRIES:
                    raise last_error
                parse_attempts += 1
                log.warn(
                    "agy_empty_stdout_retry",
                    attempt=parse_attempts,
                    stderr_head=stderr[:200],
                )
                continue

            try:
                payload = _extract_json_payload(stdout)
                output = structure.model_validate_json(payload)
            except Exception as exc:
                last_error = RuntimeError(
                    f"failed to parse agy reply as {structure.__name__}: {exc}; "
                    f"stdout head={stdout[:200]!r}"
                )
                if parse_attempts >= _MAX_PARSE_RETRIES:
                    raise last_error
                parse_attempts += 1
                log.warn(
                    "agy_parse_failed_retry",
                    attempt=parse_attempts,
                    error=str(exc)[:200],
                    stdout_head=stdout[:200],
                )
                continue

            return output, usage
