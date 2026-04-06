import asyncio
import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import logfire
from dotenv import load_dotenv

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


# Context variables for tracking IDs
current_task_id: ContextVar[str | None] = ContextVar("current_task_id", default=None)
current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)


def set_task_id(task_id: str) -> None:
    """Set the current task ID for logging context."""
    current_task_id.set(task_id)


def get_task_id() -> str | None:
    """Get the current task ID from context."""
    return current_task_id.get()


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
_original_trace = getattr(logfire, "trace", None)
_original_notice = getattr(logfire, "notice", None)
_original_fatal = getattr(logfire, "fatal", None)


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


# Patch the main logging methods
logfire.debug = _debug
logfire.info = _info
logfire.warn = _warn
logfire.error = _error

# Patch optional methods if they exist
if _original_trace:

    def _trace(msg: str, **kwargs: Any) -> Any:
        updated = _add_context_to_kwargs(**kwargs)
        _log_to_local_file("debug", msg, **updated)
        return _original_trace(msg, **updated)

    logfire.trace = _trace
if _original_notice:

    def _notice(msg: str, **kwargs: Any) -> Any:
        updated = _add_context_to_kwargs(**kwargs)
        _log_to_local_file("info", msg, **updated)
        return _original_notice(msg, **updated)

    logfire.notice = _notice
if _original_fatal:

    def _fatal(msg: str, **kwargs: Any) -> Any:
        updated = _add_context_to_kwargs(**kwargs)
        _log_to_local_file("fatal", msg, **updated)
        return _original_fatal(msg, **updated)

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
            _log_to_local_file(
                "info",
                f"span.cancelled {self._name}",
                duration_seconds=duration,
                error_type=type(exc).__name__,
                **self._attributes,
            )
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
            _log_to_local_file(
                "info",
                f"span.cancelled {self._name}",
                duration_seconds=duration,
                error_type=type(exc).__name__,
                **self._attributes,
            )
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
