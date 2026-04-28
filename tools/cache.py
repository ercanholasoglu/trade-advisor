"""
🗄️ Cache System — TTL-based function caching
===============================================
functools-based in-memory cache with TTL support.
Zero external dependencies (no Redis needed).

TTL defaults:
  - Price data: 15 minutes (900s)
  - Technical indicators: 15 minutes
  - News/sentiment: 30 minutes (1800s)
  - Fundamental data: 24 hours (86400s)
  - Market overview: 15 minutes
  - KAP disclosures: 30 minutes

Usage:
    @cache(ttl=900)
    def get_price_history(ticker, period="1mo", interval="1d"):
        ...

    # Manual clear
    cache_clear()
    cache_stats()
"""

import time
import hashlib
import json
import threading
from functools import wraps
from typing import Any

# Global cache store
_cache_store: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()
_cache_hits = 0
_cache_misses = 0

# Default TTLs by data type
TTL_PRICE = 900        # 15 min
TTL_TECHNICAL = 900    # 15 min
TTL_NEWS = 1800        # 30 min
TTL_FUNDAMENTAL = 86400  # 24 hours
TTL_MARKET = 900       # 15 min
TTL_KAP = 1800         # 30 min
TTL_CRYPTO = 900       # 15 min
TTL_RISK = 900         # 15 min
TTL_OPTIONS = 900      # 15 min
TTL_INSIDER = 3600     # 1 hour
TTL_MACRO = 3600       # 1 hour
TTL_BIST = 900         # 15 min


def _make_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Create a deterministic cache key from function name + args."""
    key_data = {
        "func": func_name,
        "args": [str(a) for a in args],
        "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def cache(ttl: int = 900):
    """
    Decorator that caches function results with TTL (time-to-live in seconds).
    
    Args:
        ttl: Cache duration in seconds. Default 900 (15 min).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global _cache_hits, _cache_misses
            key = _make_key(func.__name__, args, kwargs)

            with _cache_lock:
                if key in _cache_store:
                    expire_time, cached_value = _cache_store[key]
                    if time.time() < expire_time:
                        _cache_hits += 1
                        return cached_value
                    else:
                        # Expired — remove
                        del _cache_store[key]

            # Cache miss — call function
            _cache_misses += 1
            result = func(*args, **kwargs)

            with _cache_lock:
                _cache_store[key] = (time.time() + ttl, result)

            return result
        
        wrapper._cache_ttl = ttl
        wrapper._original = func
        return wrapper
    return decorator


def cache_clear():
    """Clear all cached data."""
    global _cache_store, _cache_hits, _cache_misses
    with _cache_lock:
        count = len(_cache_store)
        _cache_store.clear()
        _cache_hits = 0
        _cache_misses = 0
    return {"cleared": count}


def cache_stats() -> dict:
    """Return cache statistics."""
    with _cache_lock:
        total = _cache_hits + _cache_misses
        hit_rate = (_cache_hits / total * 100) if total > 0 else 0
        # Count expired entries
        now = time.time()
        active = sum(1 for exp, _ in _cache_store.values() if exp > now)
        expired = len(_cache_store) - active
        return {
            "total_entries": len(_cache_store),
            "active_entries": active,
            "expired_entries": expired,
            "cache_hits": _cache_hits,
            "cache_misses": _cache_misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }


def cache_remove(func_name: str, *args, **kwargs):
    """Remove a specific cache entry."""
    key = _make_key(func_name, args, kwargs)
    with _cache_lock:
        if key in _cache_store:
            del _cache_store[key]
            return True
    return False
