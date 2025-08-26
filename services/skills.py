import os
import re
import httpx
from typing import Optional, Tuple, Dict, Any


def extract_weather_location(query: str) -> Optional[str]:
    """Extract a location from natural queries. Returns None if no clear city is present.

    Avoids guessing from generic capitalized phrases like "What Is The" or "Right Now".
    """
    if not query:
        return None
    q = query.strip().lower()
    # Common, explicit patterns
    patterns = [
        r"weather\s+in\s+([a-z\s,]+)",
        r"temperature\s+in\s+([a-z\s,]+)",
        r"forecast\s+for\s+([a-z\s,]+)",
        r"(?:in\s+)?([a-z\s,]+)\s+weather",
    ]
    stop_trailing = re.compile(r"\b(right now|now|today|currently|outside)\b", re.I)
    for p in patterns:
        m = re.search(p, q)
        if m:
            city = m.group(1)
            # Strip trailing time words like "right now"
            city = stop_trailing.sub("", city)
            # Remove stray punctuation/commas/spaces
            city = re.sub(r"[^a-z\s,]", " ", city)
            city = re.sub(r"\s+", " ", city).strip(" ,")
            if city and len(city) >= 2:
                return city.title()
    # No safe match: do not guess from capitalization; require explicit pattern
    return None


async def fetch_weather(location: str, *, timeout: float = 8.0) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch current weather from OpenWeatherMap by city name (metric units).

    Env: OPENWEATHER_API_KEY must be set.
    Returns (data, error). If error is not None, data will be None.
    """
    if not location:
        return None, "No location provided"
    api_key = (os.getenv("OPENWEATHER_API_KEY") or "").strip('"\'')
    if not api_key:
        return None, "OPENWEATHER_API_KEY not configured"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Support inputs like "City" or "City, CountryCode"
            params = {"q": location, "appid": api_key, "units": "metric"}
            res = await client.get("https://api.openweathermap.org/data/2.5/weather", params=params)
            if res.status_code == 404:
                return None, f"City not found: '{location}'"
            if res.status_code != 200:
                return None, f"OpenWeather error: {res.status_code}"
            r = res.json() or {}
            name = (r.get("name") or location).strip()
            sys = r.get("sys") or {}
            country = sys.get("country")
            main = r.get("main") or {}
            wind = r.get("wind") or {}
            clouds = (r.get("clouds") or {}).get("all")
            weather_arr = r.get("weather") or []
            desc = weather_arr[0].get("description") if weather_arr else None
            # Convert wind speed m/s -> km/h
            wind_kmh = (wind.get("speed") or 0) * 3.6
            data = {
                "resolved_name": f"{name}, {country}" if country else name,
                "temperature_c": main.get("temp"),
                "apparent_c": main.get("feels_like"),
                "humidity": main.get("humidity"),
                "pressure_hpa": main.get("pressure"),
                "cloud_cover": clouds,
                "wind_kmh": wind_kmh,
                "wind_deg": wind.get("deg"),
                "conditions": desc.title() if desc else None,
            }
            return data, None
    except httpx.ReadTimeout:
        return None, "Weather service timed out"
    except Exception as e:
        return None, f"Weather error: {e}"


def format_weather_response(data: Dict[str, Any]) -> str:
    """Compose a concise, futuristic weather summary."""
    name = data.get("resolved_name") or "your location"
    t = data.get("temperature_c")
    feels = data.get("apparent_c")
    hum = data.get("humidity")
    clouds = data.get("cloud_cover")
    cond = data.get("conditions")
    wind = data.get("wind_kmh")
    # Build descriptors
    parts = []
    if t is not None:
        parts.append(f"{t:.1f}°C")
    if feels is not None and abs((feels or 0) - (t or 0)) >= 0.6:
        parts.append(f"feels {feels:.1f}°C")
    if hum is not None:
        parts.append(f"{hum}% humidity")
    if cond:
        parts.append(cond)
    elif clouds is not None:
        sky = ("clear" if clouds < 20 else "partly cloudy" if clouds < 60 else "overcast")
        parts.append(sky)
    if wind is not None:
        parts.append(f"winds {wind:.0f} km/h")

    summary = ", ".join(parts) if parts else "conditions unavailable"
    return f"Weather scan for {name}: {summary}."


async def handle_weather_query(query: str) -> Optional[str]:
    """High-level handler: detect location, fetch, and format a response. Returns None if not applicable."""
    loc = extract_weather_location(query or "")
    if not loc:
        # Only treat as weather if the word appears
        if "weather" not in (query or "").lower() and "temperature" not in (query or "").lower():
            return None
        return "I can fetch live weather. Say a city — e.g., 'weather in Bangalore' or 'Mumbai weather'."
    data, err = await fetch_weather(loc)
    if err:
        return f"Weather service note: {err}. Try another location."
    if not data:
        return "I couldn't fetch weather right now. Try again shortly."
    return format_weather_response(data)
