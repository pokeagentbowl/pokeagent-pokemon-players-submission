"""
Environment variable helper functions for reading and parsing configuration.
"""

import os


def get_env_int(name, default):
    """Read an integer environment variable with fallback."""
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        print(f"⚠️ Invalid value for {name}: {value!r}. Using default {default}.")
        return default


def get_env_str(name):
    """Read a string environment variable, returning None if unset or empty."""
    value = os.getenv(name)
    return value if value not in (None, "") else None


def parse_env_bool(name):
    """Parse a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    print(f"⚠️ Ignoring invalid boolean for {name}: {value!r}")
    return None
