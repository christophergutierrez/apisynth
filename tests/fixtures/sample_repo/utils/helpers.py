"""Utility helpers."""

from typing import List, Optional


def slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    return text.lower().replace(" ", "-")


def chunk_list(lst: list, size: int) -> List[list]:
    """Split a list into chunks of given size."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dicts."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def find_by_key(items: List[dict], key: str, value) -> Optional[dict]:
    """Find first dict in list where items[key] == value."""
    for item in items:
        if item.get(key) == value:
            return item
    return None


class Retry:
    """Simple retry wrapper."""

    def __init__(self, max_attempts: int = 3, delay: float = 1.0):
        self.max_attempts = max_attempts
        self.delay = delay

    def __call__(self, fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            import time
            last_exc = None
            for _ in range(self.max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    time.sleep(self.delay)
            raise last_exc

        return wrapper

    def reset(self):
        pass
