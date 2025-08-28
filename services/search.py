import httpx
from typing import Any, Dict, List, Optional, Tuple
from services.config import config as app_config


async def web_search(query: str, *, max_results: int = 5, timeout: float = 10.0) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Call Tavily search API to get an answer and sources. Returns (data, error)."""
    api_key = (app_config.get("TAVILY_API_KEY") or "").strip('\"\'')
    if not api_key:
        return None, "TAVILY_API_KEY not configured"
    if not query or not query.strip():
        return None, "Empty query"
    payload = {
        "api_key": api_key,
        "query": query,
        "include_answer": True,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_raw_content": False,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.post("https://api.tavily.com/search", json=payload)
            if res.status_code != 200:
                return None, f"Search API error: {res.status_code}"
            data = res.json() or {}
            return data, None
    except httpx.ReadTimeout:
        return None, "Search timed out"
    except Exception as e:
        return None, f"Search error: {e}"


def format_search_summary(data: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
    """Extract a concise summary and a normalized list of sources from Tavily result."""
    answer = (data.get("answer") or "").strip()
    # Normalize sources
    raw_sources = data.get("results") or []
    sources: List[Dict[str, str]] = []
    for r in raw_sources:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or url).strip()
        snippet = (r.get("content") or r.get("snippet") or "").strip()
        if url:
            sources.append({"title": title[:120], "url": url, "snippet": snippet[:240]})
    # Keep top 5
    sources = sources[:5]
    if not answer:
        # Fallback to a brief synthesis from titles
        if sources:
            answer = "Here’s a concise readout from the top results."
        else:
            answer = "I couldn’t find enough reliable information to answer that right now."
    return answer, sources


def speechify_summary(text: str) -> str:
    """Trim search summary for speech: concise and friendly."""
    t = (text or "").strip()
    if len(t) > 600:
        t = t[:560].rstrip() + ". More details are available in the sources."
    return t
