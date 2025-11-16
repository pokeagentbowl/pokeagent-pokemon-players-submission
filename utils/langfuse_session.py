"""
Helpers for building and registering Langfuse session identifiers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

DEFAULT_TEMPLATE = "{AGENT_MODE}_{AGENT_TYPE}_{DATETIME}_{AGENT_MODEL_NAME}"
_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class SessionContext:
    """Resolved values for Langfuse session templating."""

    agent_mode: str
    agent_type: str
    agent_model_name: str
    agent_backend: str
    datetime_stamp: str

    @classmethod
    def from_env(cls, overrides: Mapping[str, str] | None = None) -> "SessionContext":
        """Create a context populated from environment variables with optional overrides."""
        data = {
            "AGENT_MODE": os.getenv("AGENT_MODE", "legacy"),
            "AGENT_TYPE": os.getenv("AGENT_TYPE") or os.getenv("AGENT_SCAFFOLD", "unknown"),
            "AGENT_MODEL_NAME": os.getenv("AGENT_MODEL_NAME", "unknown"),
            "AGENT_BACKEND": os.getenv("AGENT_BACKEND")
            or os.getenv("POKEAGENT_BACKEND")
            or "unknown",
        }
        datetime_override: str | None = None
        if overrides:
            for key, value in overrides.items():
                if key and value:
                    if key == "DATETIME":
                        datetime_override = value
                    else:
                        data[key] = value

        return cls(
            agent_mode=_sanitize_segment(data.get("AGENT_MODE", "")),
            agent_type=_sanitize_segment(data.get("AGENT_TYPE", "")),
            agent_model_name=_sanitize_segment(data.get("AGENT_MODEL_NAME", "")),
            agent_backend=_sanitize_segment(data.get("AGENT_BACKEND", "")),
            datetime_stamp=_sanitize_segment(datetime_override or datetime.utcnow().strftime("%Y%m%d_%H%M%S")),
        )

    def as_mapping(self) -> dict[str, str]:
        """Expose the context as a mapping for string formatting."""
        return {
            "AGENT_MODE": self.agent_mode,
            "AGENT_TYPE": self.agent_type,
            "AGENT_MODEL_NAME": self.agent_model_name,
            "AGENT_BACKEND": self.agent_backend,
            "DATETIME": self.datetime_stamp,
            # Backward compatibility
            "AGENT_SCAFFOLD": self.agent_type,  # Map old name to new
        }


class _SafeDict(dict[str, str]):
    """Provide default values for missing placeholders during format_map."""

    def __missing__(self, key: str) -> str:
        return "unknown"


def _sanitize_segment(value: str) -> str:
    """Normalize input into a Langfuse-friendly token."""
    cleaned = _SANITIZE_PATTERN.sub("-", value.strip())
    collapsed = re.sub(r"-+", "-", cleaned).strip("-_")
    return collapsed or "unknown"


def compute_langfuse_session_id(overrides: Mapping[str, str] | None = None) -> str:
    """
    Build a session identifier from environment or provided overrides.

    The template is sourced from LANGFUSE_SESSION_ID if present, otherwise the
    default of {AGENT_MODE}_{AGENT_TYPE}_{DATETIME}_{AGENT_MODEL_NAME} is used.
    
    For backward compatibility, {AGENT_SCAFFOLD} is still supported and maps to AGENT_TYPE.
    """
    template = os.getenv("LANGFUSE_SESSION_ID") or DEFAULT_TEMPLATE
    context = SessionContext.from_env(overrides)

    rendered = template.format_map(_SafeDict(context.as_mapping()))
    condensed = re.sub(r"-{2,}", "-", rendered)
    trimmed = condensed.strip("-_")
    return trimmed or "session"


def initialize_langfuse_session(overrides: Mapping[str, str] | None = None, force: bool = False) -> str:
    """
    Compute and register the Langfuse session ID for the current process.

    If force is False, the function respects any session that is already set.
    """
    from utils.vlm import get_langfuse_session_id, set_langfuse_session_id  # Local import to avoid cycles

    if not force:
        resolved_env = os.getenv("LANGFUSE_SESSION_ID_RESOLVED")
        if resolved_env:
            set_langfuse_session_id(resolved_env)
            return resolved_env
        existing = get_langfuse_session_id()
        if existing:
            return existing

    session_id = compute_langfuse_session_id(overrides)
    set_langfuse_session_id(session_id)
    os.environ["LANGFUSE_SESSION_ID_RESOLVED"] = session_id
    return session_id
