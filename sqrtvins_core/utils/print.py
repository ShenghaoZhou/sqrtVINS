"""
Logging — Python port of ov_core/src/utils/print.h and colors.h.

The C++ side exposes `PRINT_ALL` / `PRINT_DEBUG` / `PRINT_INFO` /
`PRINT_WARNING` / `PRINT_ERROR` macros that funnel into
`ov_core::Printer::debugPrint(level, __FILE__, TOSTRING(__LINE__), fmt, ...)`.
The threshold is a global `current_print_level`; a message prints iff its level
is >= the threshold.

Python's `logging` does this with less code and is stream-aware, so we wrap it
rather than re-implementing a level counter. The one behaviour that matters to
the ports and that `logging` does not give you for free is that the C++ macros
accept `printf`-style format strings with the `%` converted to a literal `%`
(see `parser->parse_config` callers that do
`PRINT_DEBUG("  - num cameras: %d\n", n)`). Those call sites become plain f-strings
in Python, so no shim is needed.

The ANSI color macros from `colors.h` are kept as plain string constants —
`sqrtvins_core/utils/print.py` is the module the ports import them from.
"""

from __future__ import annotations

import logging
import sys

# ---------------------------------------------------------------------------
# colors.h — ANSI escape codes, verbatim
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
REDPURPLE = "\033[95m"

BOLDBLACK = "\033[1m\033[30m"
BOLDRED = "\033[1m\033[31m"
BOLDGREEN = "\033[1m\033[32m"
BOLDYELLOW = "\033[1m\033[33m"
BOLDBLUE = "\033[1m\033[34m"
BOLDMAGENTA = "\033[1m\033[35m"
BOLDCYAN = "\033[1m\033[36m"
BOLDWHITE = "\033[1m\033[37m"
BOLDREDPURPLE = "\033[1m\033[95m"


# ---------------------------------------------------------------------------
# PrintLevel — the ordering is meaningful: a lower value is more verbose, and
# `set_print_level` is what the config loader calls to silence startup noise.
# ---------------------------------------------------------------------------

ALL = 0
DEBUG = 1
INFO = 2
WARNING = 3
ERROR = 4
SILENT = 5

_LEVEL_NAMES = {
    "ALL": ALL, "DEBUG": DEBUG, "INFO": INFO,
    "WARNING": WARNING, "ERROR": ERROR, "SILENT": SILENT,
}

# Python logging level for each C++ level. SILENT maps to CRITICAL+1 so nothing
# gets through; logging has no level 6, so we clamp at CRITICAL and gate below.
_LOGLEVEL = {
    ALL: logging.DEBUG,
    DEBUG: logging.DEBUG,
    INFO: logging.INFO,
    WARNING: logging.WARNING,
    ERROR: logging.ERROR,
    SILENT: logging.CRITICAL,
}

_logger = logging.getLogger("sqrtvins")
if not _logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_h)
_logger.setLevel(logging.DEBUG)

current_print_level: int = ALL


def set_print_level(level) -> None:
    """Port of `Printer::setPrintLevel`. Accepts a name or a PrintLevel int."""
    global current_print_level
    if isinstance(level, str):
        try:
            level = _LEVEL_NAMES[level.upper()]
        except KeyError as exc:
            raise ValueError(
                f"unknown print level {level!r}; valid: {sorted(_LEVEL_NAMES)}"
            ) from exc
    current_print_level = level
    _logger.setLevel(_LOGLEVEL[level])


def debug_print(level: int, message: str) -> None:
    """One entry point behind every macro. Mirrors `Printer::debugPrint`."""
    if level < current_print_level:
        return
    _logger.log(_LOGLEVEL[level], message)


def print_all(*message: object) -> None:
    debug_print(ALL, " ".join(str(m) for m in message))


def print_debug(*message: object) -> None:
    debug_print(DEBUG, " ".join(str(m) for m in message))


def print_info(*message: object) -> None:
    debug_print(INFO, " ".join(str(m) for m in message))


def print_warning(*message: object) -> None:
    debug_print(WARNING, " ".join(str(m) for m in message))


def print_error(*message: object) -> None:
    debug_print(ERROR, " ".join(str(m) for m in message))
