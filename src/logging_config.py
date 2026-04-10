import asyncio
import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import logfire
from dotenv import load_dotenv

from src.llms.models import TokenUsage

# Load environment variables
load_dotenv()

# If set to "1", do not send logs to Logfire; only write locally
LOCAL_LOGS_ONLY = os.environ.get("LOCAL_LOGS_ONLY", "0") == "1"

# Suppress OpenTelemetry export errors
logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.ERROR)
logging.getLogger("opentelemetry.exporter.otlp").setLevel(logging.ERROR)


def scrubbing_callback(m: logfire.ScrubMatch):
    # Disable all scrubbing - return the original value for everything
    # This lets us see full error messages including auth errors, cookies, etc.
    return m.value


# Initialize Logfire with the API key from env
logfire.configure(
    token=os.environ.get("LOGFIRE_API_KEY"),
    service_name="arc-lang",
    send_to_logfire=not LOCAL_LOGS_ONLY,
    console=False,  # Disable console logging,
    scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback),
)

# -----------------------------------------------------------------------------
# Local file logging setup
# -----------------------------------------------------------------------------


LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

local_logger = logging.getLogger("arc_local")
local_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))


def configure_local_log_path(log_file: Path) -> None:
    """Point the rotating file handler at log_file (e.g. results/<stamp>/arc.log)."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    local_logger.addHandler(file_handler)


def _serialize_kwargs_for_log(kwargs: dict[str, Any]) -> str:
    if not kwargs:
        return ""

    def default_serializer(o: Any) -> str:
        try:
            return repr(o)
        except Exception:  # noqa: BLE001
            return str(o)

    try:
        return json.dumps(
            kwargs,
            ensure_ascii=False,
            separators=(",", ":"),
            default=default_serializer,
        )
    except Exception:  # noqa: BLE001
        # Last-resort fallback
        return str({k: str(v) for k, v in kwargs.items()})


def _log_to_local_file(level: str, message: str, **kwargs: Any) -> None:
    structured_part = _serialize_kwargs_for_log(kwargs)
    final_message = message if not structured_part else f"{message} | {structured_part}"
    if level == "debug":
        local_logger.debug(final_message)
    elif level == "info":
        local_logger.info(final_message)
    elif level in ("warn", "warning"):
        local_logger.warning(final_message)
    elif level == "error":
        local_logger.error(final_message)
    elif level == "fatal":
        local_logger.critical(final_message)
    else:
        local_logger.info(final_message)


@dataclass
class TaskLogContext:
    task_id: str
    cumulative: TokenUsage
    max_single_call_total_tokens: int = 0


# Per challenge asyncio task: task_id + in-place TokenUsage rollup + peak single-call total
current_task_log: ContextVar[TaskLogContext | None] = ContextVar(
    "current_task_log", default=None
)
current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)


def set_task_id(task_id: str) -> None:
    """Set task id and fresh token rollup state for this challenge."""
    current_task_log.set(TaskLogContext(task_id=task_id, cumulative=TokenUsage()))


def get_task_id() -> str | None:
    """Get the current task ID from context."""
    ctx = current_task_log.get()
    return ctx.task_id if ctx else None


def record_llm_token_usage(usage: TokenUsage) -> None:
    ctx = current_task_log.get()
    if ctx is None:
        return
    ctx.cumulative += usage
    ct = usage.total_tokens
    if ct > ctx.max_single_call_total_tokens:
        ctx.max_single_call_total_tokens = ct


def get_challenge_token_totals() -> tuple[TokenUsage, int]:
    ctx = current_task_log.get()
    if ctx is None:
        return TokenUsage(), 0
    return ctx.cumulative, ctx.max_single_call_total_tokens


def merge_run_token_usage(
    calls: list[tuple[TokenUsage, int]],
) -> tuple[TokenUsage, int]:
    total_usage = TokenUsage()
    max_tokens = 0
    for usage, max_tokens_in_call in calls:
        total_usage += usage
        max_tokens = max(max_tokens, max_tokens_in_call)
    return total_usage, max_tokens


def set_run_id(run_id: str) -> None:
    """Set the current run ID for logging context."""
    current_run_id.set(run_id)


def get_run_id() -> str | None:
    """Get the current run ID from context."""
    return current_run_id.get()


def generate_run_id() -> str:
    """Generate a new run ID and set it in context."""
    run_id = str(uuid.uuid4())[:8]  # Use first 8 chars for readability
    set_run_id(run_id)
    return run_id


# Store original methods
_original_debug = logfire.debug
_original_info = logfire.info
_original_warn = logfire.warn
_original_error = logfire.error
_original_trace = logfire.trace
_original_notice = logfire.notice
_original_fatal = logfire.fatal


def _add_context_to_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Add context variables to kwargs."""
    task_id = get_task_id()
    if task_id:
        kwargs["task_id"] = task_id

    run_id = get_run_id()
    if run_id:
        kwargs["run_id"] = run_id

    return kwargs


def _debug(msg: str, **kwargs: Any) -> Any:
    updated = _add_context_to_kwargs(**kwargs)
    _log_to_local_file("debug", msg, **updated)
    return _original_debug(msg, **updated)


def _info(msg: str, **kwargs: Any) -> Any:
    updated = _add_context_to_kwargs(**kwargs)
    _log_to_local_file("info", msg, **updated)
    return _original_info(msg, **updated)


def _warn(msg: str, **kwargs: Any) -> Any:
    updated = _add_context_to_kwargs(**kwargs)
    _log_to_local_file("warn", msg, **updated)
    return _original_warn(msg, **updated)


def _error(msg: str, **kwargs: Any) -> Any:
    updated = _add_context_to_kwargs(**kwargs)
    _log_to_local_file("error", msg, **updated)
    return _original_error(msg, **updated)


def _trace(msg: str, **kwargs: Any) -> Any:
    updated = _add_context_to_kwargs(**kwargs)
    _log_to_local_file("debug", msg, **updated)
    return _original_trace(msg, **updated)


def _notice(msg: str, **kwargs: Any) -> Any:
    updated = _add_context_to_kwargs(**kwargs)
    _log_to_local_file("info", msg, **updated)
    return _original_notice(msg, **updated)


def _fatal(msg: str, **kwargs: Any) -> Any:
    updated = _add_context_to_kwargs(**kwargs)
    _log_to_local_file("fatal", msg, **updated)
    return _original_fatal(msg, **updated)


# Patch the main logging methods
logfire.debug = _debug
logfire.info = _info
logfire.warn = _warn
logfire.error = _error
logfire.trace = _trace
logfire.notice = _notice
logfire.fatal = _fatal


# Also patch span to include context
_original_span = logfire.span


class _LocalSpanWrapper:
    def __init__(self, inner_cm: Any, name: str, attributes: dict[str, Any]):
        self._inner_cm = inner_cm
        self._name = name
        self._attributes = attributes
        self._start_time: float | None = None

    # Sync context manager
    def __enter__(self):
        self._start_time = time.time()
        _log_to_local_file("info", f"span.start {self._name}", **self._attributes)
        return self._inner_cm.__enter__()

    def __exit__(self, exc_type, exc, tb):
        duration = (
            (time.time() - self._start_time) if self._start_time is not None else None
        )
        if isinstance(exc, asyncio.CancelledError):
            pass
        elif exc is not None:
            _log_to_local_file(
                "error",
                f"span.error {self._name}",
                duration_seconds=duration,
                error=str(exc),
                error_type=type(exc).__name__,
                **self._attributes,
            )
        else:
            _log_to_local_file(
                "info",
                f"span.end {self._name}",
                duration_seconds=duration,
                **self._attributes,
            )
        return self._inner_cm.__exit__(exc_type, exc, tb)

    # Async context manager
    async def __aenter__(self):
        self._start_time = time.time()
        _log_to_local_file("info", f"span.start {self._name}", **self._attributes)
        if hasattr(self._inner_cm, "__aenter__"):
            return await self._inner_cm.__aenter__()
        return self._inner_cm.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        duration = (
            (time.time() - self._start_time) if self._start_time is not None else None
        )
        if isinstance(exc, asyncio.CancelledError):
            pass
        elif exc is not None:
            _log_to_local_file(
                "error",
                f"span.error {self._name}",
                duration_seconds=duration,
                error=str(exc),
                error_type=type(exc).__name__,
                **self._attributes,
            )
        else:
            _log_to_local_file(
                "info",
                f"span.end {self._name}",
                duration_seconds=duration,
                **self._attributes,
            )
        if hasattr(self._inner_cm, "__aexit__"):
            return await self._inner_cm.__aexit__(exc_type, exc, tb)
        return self._inner_cm.__exit__(exc_type, exc, tb)


def _span_with_context(name: str, **kwargs: Any) -> Any:
    """Wrapper that adds context and logs span lifecycle locally."""
    task_id = get_task_id()
    if task_id:
        kwargs["task_id"] = task_id

    run_id = get_run_id()
    if run_id:
        kwargs["run_id"] = run_id

    inner = _original_span(name, **kwargs)
    return _LocalSpanWrapper(inner, name, kwargs)


logfire.span = _span_with_context
