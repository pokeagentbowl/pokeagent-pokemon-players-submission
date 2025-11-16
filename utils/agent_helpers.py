"""
Utility helpers used by the agent runtime.

Currently provides helpers for synchronising local LLM metrics with the
FastAPI server so the web UI can display up-to-date token/cost counters.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests

from utils.llm_logger import get_llm_logger

logger = logging.getLogger(__name__)

# Cache the computed base URL so we don't rebuild it every step.
_SERVER_BASE_URL: Optional[str] = None


def _ensure_scheme(host: str) -> str:
    """Ensure the host string includes a scheme."""
    if host.startswith(("http://", "https://")):
        return host
    return f"http://{host}"


def _build_server_base_url() -> str:
    """
    Determine the server base URL from environment variables.

    Priority:
        1. SERVER_URL or AGENT_SERVER_URL (full URL)
        2. SERVER_HOST / AGENT_SERVER_HOST + SERVER_PORT / AGENT_PORT
        3. Default to http://127.0.0.1:8000
    """
    url = os.getenv("SERVER_URL") or os.getenv("AGENT_SERVER_URL")
    if url:
        return url.rstrip("/")

    host = (
        os.getenv("SERVER_HOST")
        or os.getenv("AGENT_SERVER_HOST")
        or os.getenv("POKEAGENT_SERVER_HOST")
        or "127.0.0.1"
    )
    host = _ensure_scheme(host.strip())

    # If host already includes a port, respect it.
    host_without_scheme = host.split("://", 1)[-1]
    if ":" in host_without_scheme:
        return host.rstrip("/")

    port = (
        os.getenv("SERVER_PORT")
        or os.getenv("AGENT_PORT")
        or os.getenv("POKEAGENT_SERVER_PORT")
        or "8000"
    )
    return f"{host.rstrip('/')}:{port}"


def get_server_base_url() -> str:
    """Public accessor for the resolved server base URL."""
    global _SERVER_BASE_URL
    if not _SERVER_BASE_URL:
        _SERVER_BASE_URL = _build_server_base_url()
    return _SERVER_BASE_URL


def update_server_metrics(metrics: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Push the latest cumulative LLM metrics to the server.

    Called by the agent after each successful step so the server can keep its
    `/metrics` endpoint and stream UI in sync.

    Args:
        metrics: Optional metrics dictionary. If omitted, the current totals
                 are pulled from the global LLM logger.

    Returns:
        Optional response JSON from the server, or None if the update failed.
    """
    if os.getenv("DISABLE_SERVER_METRICS_UPLOAD", "").lower() in ("1", "true"):
        return None

    if metrics is None:
        try:
            llm_logger = get_llm_logger()
            metrics = llm_logger.get_cumulative_metrics()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not gather metrics for server sync: %s", exc)
            metrics = None

    payload = {}
    if metrics:
        # Ensure start_time is serialisable (it is a float normally)
        payload["metrics"] = metrics

    try:
        response = requests.post(
            f"{get_server_base_url()}/agent_step",
            json=payload or None,
            timeout=float(os.getenv("METRICS_SYNC_TIMEOUT", "2.0")),
        )
        response.raise_for_status()
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Metrics sync failed: %s", exc)

    return None


__all__ = ["get_server_base_url", "update_server_metrics"]
