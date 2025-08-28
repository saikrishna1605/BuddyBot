import os
import threading
from typing import Dict, Optional, Any

import assemblyai as aai
import google.generativeai as genai


class _AppConfig:
    """Thread-safe runtime configuration for API keys and model settings.

    Prefers user-provided values set at runtime; falls back to environment variables.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._values: Dict[str, Any] = {
            # Keys
            "ASSEMBLYAI_API_KEY": None,
            "GEMINI_API_KEY": None,
            "MURF_API_KEY": None,
            "OPENWEATHER_API_KEY": None,
            "TAVILY_API_KEY": None,
            # Model settings
            "LLM_MODEL": os.getenv("LLM_MODEL", "gemini-1.5-flash"),
            "LLM_TEMPERATURE": float(os.getenv("LLM_TEMPERATURE", "0.6")),
        }

    def init_from_env(self) -> None:
        with self._lock:
            for k in [
                "ASSEMBLYAI_API_KEY",
                "GEMINI_API_KEY",
                "MURF_API_KEY",
                "OPENWEATHER_API_KEY",
                "TAVILY_API_KEY",
            ]:
                env_v = os.getenv(k)
                if env_v:
                    self._values[k] = env_v

    def set_many(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Update multiple config values; returns which SDKs were reconfigured."""
        changed = {}
        with self._lock:
            for key, val in (data or {}).items():
                if key not in self._values:
                    continue
                self._values[key] = val
            # Apply to SDKs if present
            changed = self._apply_sdks_locked()
        return changed

    def _apply_sdks_locked(self) -> Dict[str, bool]:
        out = {"assemblyai": False, "gemini": False}
        akey = self._values.get("ASSEMBLYAI_API_KEY")
        if akey:
            try:
                aai.settings.api_key = str(akey).strip("\"' ")
                out["assemblyai"] = True
            except Exception:
                pass
        gkey = self._values.get("GEMINI_API_KEY")
        if gkey:
            try:
                genai.configure(api_key=str(gkey).strip("\"' "))
                out["gemini"] = True
            except Exception:
                pass
        return out

    def apply_sdks(self) -> Dict[str, bool]:
        with self._lock:
            return self._apply_sdks_locked()

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        with self._lock:
            v = self._values.get(key)
            if v is None:
                # Fall back to env lazily
                env_v = os.getenv(key)
                return env_v if env_v is not None else default
            return v

    def get_all_masked(self) -> Dict[str, Any]:
        with self._lock:
            out: Dict[str, Any] = {}
            for k, v in self._values.items():
                if k.endswith("_API_KEY"):
                    if v and isinstance(v, str) and len(v) > 8:
                        out[k] = v[:4] + "…" + v[-4:]
                    elif v:
                        out[k] = "•••"
                    else:
                        out[k] = ""
                else:
                    out[k] = v
            return out


# Singleton instance
config = _AppConfig()
