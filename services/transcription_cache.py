from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import List, Dict

# Simple in-memory cache guarded by a lock
_recent: List[Dict[str, str]] = []
_lock = Lock()
_MAX = 10

def add_transcription_to_cache(text: str, session_id: str) -> None:
    """Add a transcription to the shared recent cache."""
    global _recent
    item = {
        "text": text,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
    }
    with _lock:
        _recent.append(item)
        if len(_recent) > _MAX:
            _recent = _recent[-_MAX:]

def get_recent_transcriptions() -> List[Dict[str, str]]:
    """Return a copy of recent transcriptions."""
    with _lock:
        return list(_recent)
