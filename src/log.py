"""Centralized logger facade.

Import this instead of `logfire` at call sites:

    from src.log import log
    log.info("Hello", key="value")
    with log.span("work"):
        ...

This ensures `src.logging_config` is loaded (patches + local file mirroring),
and then re-exports the `logfire` API under the name `log`.
"""

# Ensure side-effects (patching + local file handler) are applied
import logfire as _logfire

import src.logging_config  # noqa: F401

# Re-export the logfire API via `log` for ergonomic imports
log = _logfire

# Optionally export top-level callables for direct import if preferred
debug = _logfire.debug
info = _logfire.info
warn = _logfire.warn
warning = _logfire.warn
error = _logfire.error
span = _logfire.span

# Best-effort optional methods
notice = getattr(_logfire, "notice", _logfire.info)
fatal = getattr(_logfire, "fatal", _logfire.error)
