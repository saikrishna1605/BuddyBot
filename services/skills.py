import re
import httpx
import math
from typing import Optional, Tuple, Dict, Any, List
from services.config import config as app_config


def extract_weather_location(query: str) -> Optional[str]:
    if not query:
        return None
    q = query.strip().lower()
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
            city = stop_trailing.sub("", city)
            city = re.sub(r"[^a-z\s,]", " ", city)
            city = re.sub(r"\s+", " ", city).strip(" ,")
            if city and len(city) >= 2:
                return city.title()
    return None


async def fetch_weather(location: str, *, timeout: float = 8.0) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not location:
        return None, "No location provided"
    api_key = (app_config.get("OPENWEATHER_API_KEY") or "").strip('"\'')
    if not api_key:
        return None, "OPENWEATHER_API_KEY not configured"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
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
    name = data.get("resolved_name") or "your location"
    t = data.get("temperature_c")
    feels = data.get("apparent_c")
    hum = data.get("humidity")
    clouds = data.get("cloud_cover")
    cond = data.get("conditions")
    wind = data.get("wind_kmh")
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
    loc = extract_weather_location(query or "")
    if not loc:
        if "weather" not in (query or "").lower() and "temperature" not in (query or "").lower():
            return None
        return "I can fetch live weather. Say a city — e.g., 'weather in Bangalore' or 'Mumbai weather'."
    data, err = await fetch_weather(loc)
    if err:
        return f"Weather service note: {err}. Try another location."
    if not data:
        return "I couldn't fetch weather right now. Try again shortly."
    return format_weather_response(data)


def _extract_ticker_candidates(query: str) -> List[str]:
    
    if not query:
        return []
    q = (query or "").strip()
    pattern = re.compile(r"\b([A-Za-z]{1,6}(?:\.(?:NS|ns|BO|bo))?)\b")
    near = re.findall(r"(?:stock|share|price|quote)\s+(of\s+)?([A-Za-z.]{1,10})", q, re.I)
    cands = []
    for _, sym in near:
        cands.append(sym)
    for m in pattern.finditer(q):
        cands.append(m.group(1))
    norm: List[str] = []
    STOP = {
        "PRICE","STOCK","SHARE","QUOTE","WHAT","SHOW","OF","THE","IN","ON","IS","AND","TO","FROM","PLEASE","TELL","ME","CHECK","CURRENT","TODAY","NOW","ABOUT","A","AN"
    }
    for c in cands:
        c = c.strip().upper()
        if c in STOP:
            continue
        if len(c) < 1:
            continue
        norm.append(c)
    seen = set()
    out = []
    for s in norm:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:3]


async def _get_inr_rate(*, timeout: float = 6.0) -> float:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get("https://open.er-api.com/v6/latest/USD")
            if r.status_code == 200:
                data = r.json() or {}
                rates = data.get("rates") or {}
                inr = rates.get("INR")
                if inr and isinstance(inr, (int, float)) and inr > 0:
                    return float(inr)
    except Exception:
        pass
    return 83.0


async def _fetch_stock_price_yf(ticker: str, *, timeout: float = 8.0) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        import yfinance as yf
    except Exception as e:
        return None, f"yfinance not installed: {e}"

    try:
        tk = yf.Ticker(ticker)
        price = None
        currency = None
        exchange = None
        try:
            info = getattr(tk, "fast_info", None)
            if info is not None:
                price = getattr(info, "last_price", None)
                currency = getattr(info, "currency", None)
                exchange = getattr(info, "exchange", None)
        except Exception:
            price = None
        if price is None:
            try:
                hist = tk.history(period="1d")
                if hist is not None and not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            except Exception:
                pass
        if price is None:
            try:
                hist = tk.history(period="5d")
                if hist is not None and not hist.empty:
                    price = float(hist["Close"].dropna().iloc[-1])
            except Exception:
                pass
        if price is None:
            return None, "No price available"
        return {"ticker": ticker.upper(), "price": float(price), "currency": (currency or "USD"), "exchange": exchange}, None
    except Exception as e:
        return None, f"Stock fetch error: {e}"


_COMPANY_MAP = {
    "apple": "AAPL",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "meta": "META",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "wipro": "WIPRO.NS",
    "hdfc": "HDFCBANK.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS",
    "hcl": "HCLTECH.NS",
}


async def _search_ticker_yahoo(query: str, *, timeout: float = 6.0) -> Optional[str]:
    q = (query or "").strip()
    if q.startswith("$") and len(q) > 1:
        q = q[1:]
    if not q:
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": q, "quotesCount": 5, "newsCount": 0},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if r.status_code != 200:
                return None
            data = r.json() or {}
            quotes = data.get("quotes") or []
            equities = [it for it in quotes if (it.get("quoteType") or "").upper() == "EQUITY" and it.get("symbol")]
            if equities:
                lower = q.lower()
                def score(it):
                    sym = (it.get("symbol") or "").upper()
                    exch = (it.get("exchange") or "").upper()
                    s = 0
                    if sym.endswith(".NS") or sym.endswith(".BO"):
                        s += 2 if any(k in lower for k in ["india","nse","bse","reliance","tcs","infosys","wipro","hdfc","icici","hcl"]) else 0
                    if exch in ("NMS","NYQ"):
                        s += 2 if any(k in lower for k in ["apple","google","alphabet","microsoft","amazon","tesla","nvidia","meta"]) else 0
                    if sym in ("AAPL","GOOGL","MSFT","AMZN","TSLA","NVDA","META"):
                        s += 3
                    return s
                equities.sort(key=score, reverse=True)
                return str(equities[0]["symbol"]).upper()
            for item in quotes:
                if item.get("symbol"):
                    return str(item["symbol"]).upper()
    except Exception:
        return None
    return None


def _remember_ticker(chat_history: Optional[List[Dict[str, Any]]], found: List[str]) -> Optional[str]:
    if found:
        return found[0]
    if not chat_history:
        return None
    for msg in reversed(chat_history[-8:]):
        content = (getattr(msg, "content", None) or msg.get("content") if isinstance(msg, dict) else None) or ""
        cands = _extract_ticker_candidates(content)
        if cands:
            return cands[0]
    return None


def _format_stock_response(data: Dict[str, Any], inr_rate: float) -> str:
    price = data.get("price")
    currency = (data.get("currency") or "USD").upper()
    ticker = data.get("ticker") or ""
    ex = data.get("exchange") or ""
    if currency != "INR":
        inr_price = price * inr_rate if (price is not None and inr_rate > 0) else None
    else:
        inr_price = price
    if inr_price is None:
        return f"{ticker}: price unavailable."
    inr_display = f"₹{inr_price:,.2f}"
    return f"{ticker} current price: {inr_display}."


async def handle_stock_query(query: str, chat_history: Optional[List[Any]] = None) -> Optional[str]:
    q = (query or "").strip()
    if not q:
        return None
    lower = q.lower()
    if not ("stock" in lower or "share" in lower or "price" in lower or "quote" in lower):
        return None

    cands = _extract_ticker_candidates(q)
    mem_ticker = _remember_ticker(chat_history, cands)
    ticker = None
    for c in cands:
        if re.search(r"\.(?:NS|BO)$", c, re.I):
            ticker = c
            break
    if not ticker:
        for name, sym in _COMPANY_MAP.items():
            if name in lower:
                ticker = sym
                break
    if not ticker and mem_ticker:
        ticker = mem_ticker
    if not ticker:
        ticker = await _search_ticker_yahoo(q)
    if not ticker and cands:
        ticker = await _search_ticker_yahoo(cands[0])
    if not ticker:
        return "Tell me the ticker symbol — e.g., 'price of AAPL', 'RELIANCE.NS stock'."
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z]{1,10}(?:\.(?:NS|BO))?", ticker):
        t2 = re.sub(r"[^A-Za-z\.]+", "", ticker)
        if re.fullmatch(r"[A-Z]{1,10}(?:\.(?:NS|BO))?", t2):
            ticker = t2
        else:
            t3 = await _search_ticker_yahoo(q)
            if t3:
                ticker = t3
            else:
                return "Tell me the ticker symbol — e.g., 'price of AAPL', 'RELIANCE.NS stock'."

    data, err = await _fetch_stock_price_yf(ticker)
    if err:
        return f"Stock service note: {err}. Try another symbol."
    if not data:
        return "I couldn't fetch that stock right now. Try again shortly."

    inr = await _get_inr_rate()
    text = _format_stock_response(data, inr)
    return text
