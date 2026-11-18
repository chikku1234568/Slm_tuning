"""Tiny client for the vLLM server (OpenAI-compatible API).

    from client import parse, parse_batch
    parse("NACH-DR-BAJAJFIN-1234567890")          # -> dict with json + timings
    parse_batch(["UPI/P2M/...", "ATM WDL/..."])   # -> list of dicts + throughput

Every result looks like:
    {"text": ..., "json": {...} or None, "raw": "<model text>",
     "valid_json": bool, "schema_ok": bool, "latency_ms": float,
     "backend": "vllm" | "ollama" | "regex_fallback", "error": None | str}

Config (env vars):
    SLM_URL     server base URL, e.g. http://localhost:8000, https://xxx.trycloudflare.com,
                or http://localhost:11434 for Ollama (CPU laptop path: serving/cpu_ollama/)
    SLM_MODEL   served model name (default "finai", set by --served-model-name)
    SLM_GUIDED  how to ask vLLM for schema-constrained output:
                  "structured_outputs" (default; current vLLM, >= 0.12)
                  "guided_json"        (older vLLM < 0.12; removed in 0.12)
                  "response_format"    (OpenAI-style json_schema; also supported by vLLM)
                  "ollama"             (Ollama / llama.cpp server: OpenAI-style response_format json_schema,
                                        which Ollama turns into a grammar; see serving/cpu_ollama/README.md)
                  "none"               (no constraint; rely on the fine-tune + validation)
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import jsonschema

try:  # works both as `python serving/client.py` and `from serving.client import ...`
    from schema import SCHEMA, SYS
except ImportError:  # pragma: no cover
    from serving.schema import SCHEMA, SYS

SLM_URL = os.environ.get("SLM_URL", "http://localhost:8000")
SLM_MODEL = os.environ.get("SLM_MODEL", "finai")
SLM_GUIDED = os.environ.get("SLM_GUIDED", "structured_outputs")
VALIDATOR = jsonschema.Draft7Validator(SCHEMA)


def _extra_body(mode):
    """Build the vLLM-specific request fields for structured (guided) decoding.

    Guided decoding = at each decode step vLLM masks out tokens that would break the
    JSON schema, so the output is *guaranteed* to parse. Small models benefit a lot.
    """
    if mode == "structured_outputs":
        return {"structured_outputs": {"json": SCHEMA}}
    if mode == "guided_json":  # legacy alias (vLLM < 0.12)
        return {"guided_json": SCHEMA}
    return None


def _client(url=None, timeout=30):
    from openai import OpenAI  # imported lazily so the fallback works without openai installed
    # vLLM ignores the API key unless started with --api-key; "x" is a placeholder.
    return OpenAI(base_url=(url or SLM_URL).rstrip("/") + "/v1", api_key=os.environ.get("SLM_API_KEY", "x"),
                  timeout=timeout, max_retries=0)


def _check(obj):
    """Return (valid_json, schema_ok) for a parsed object (or None)."""
    if obj is None:
        return False, False
    return True, VALIDATOR.is_valid(obj)


def _to_json(s):
    """Parse model text; tolerate stray text around the {...} (common without guided decoding)."""
    try:
        return json.loads(s[s.find("{"):s.rfind("}") + 1])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Offline fallback: a tiny regex heuristic (same idea as rule_tag in the notebook).
# Used only when the server is unreachable, so the demo never shows a blank screen.
# It is NOT the model; results are marked backend="regex_fallback".
# ---------------------------------------------------------------------------
_LEN = [("BAJAJFIN", "Bajaj Finance"), ("HDFCBANK LN", "HDFC Bank"), ("TATACAP", "Tata Capital"), ("IDFCFIRST", "IDFC First Bank")]


def _lab(cat, sub, cp, cpt, emi=False, ln=None, bnc=False, rev=False, pp="NA", conf=0.8):
    return dict(category=cat, sub_category=sub, counterparty_name=cp, counterparty_type=cpt, is_emi=emi,
                lender_name=ln, is_bounce_or_return=bnc, is_reversal=rev, p2p_or_p2m=pp, confidence=conf)


def _upi_name(t):
    parts = [p.strip() for p in re.split(r"[/-]", t) if p.strip()]
    names = [p for p in parts if re.fullmatch(r"[A-Za-z][A-Za-z .&]{2,}", p)
             and p.upper() not in {"UPI", "P2M", "P2A", "IMPS", "MMT", "NEFT", "CR", "PAYMENT FROM PHONE"}]
    return names[0] if names else None


def regex_fallback(text):
    u = text.upper()
    uc = u.replace(" ", "")
    lend = next((n for c, n in _LEN if c.replace(" ", "")[:4] in uc), None)
    if "RTN" in u or "RETURN" in u:
        return _lab("charges", "emi_bounce_charge", lend, "lender", emi=True, ln=lend, bnc=True)
    if u.startswith("REV") or "REVERSAL" in u:
        return _lab("refund", "imps_reversal", None, "bank", rev=True)
    if lend and any(k in u for k in ("NACH", "ECS", "EMI", "MANDATE", "ACH")):
        return _lab("loan", "emi_payment", lend, "lender", emi=True, ln=lend)
    if "ATM" in u:
        return _lab("cash", "atm_withdrawal", None, "bank")
    if "SAL" in u:
        m = re.search(r"([A-Z][A-Z ]+(?:LTD|PVT LTD|LIMITED))", u)
        return _lab("income", "salary", m.group(1).strip() if m else None, "employer")
    if "P2M" in u:
        return _lab("expense", "merchant_payment", _upi_name(text), "merchant", pp="P2M")
    if "P2A" in u or "IMPS" in u:
        return _lab("transfer", "p2p_transfer", _upi_name(text), "individual", pp="P2P")
    return _lab("other", "unknown", None, "unknown", conf=0.3)


def _fallback_result(text, err, t0):
    obj = regex_fallback(text)
    ok, sch = _check(obj)
    return {"text": text, "json": obj, "raw": json.dumps(obj), "valid_json": ok, "schema_ok": sch,
            "latency_ms": (time.perf_counter() - t0) * 1000, "backend": "regex_fallback", "error": err}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse(text, url=None, model=None, guided=None, cli=None, timeout=30):
    """Send ONE narration line to the server and return a result dict (see module docstring)."""
    t0 = time.perf_counter()
    guided = guided or SLM_GUIDED
    kwargs = dict(
        model=model or SLM_MODEL,
        messages=[{"role": "system", "content": SYS},   # identical every call -> prefix-cache hit
                  {"role": "user", "content": text}],
        temperature=0,      # greedy: deterministic, best for extraction
        max_tokens=200,     # our JSON is ~60-100 tokens; caps decode time and KV-cache use
    )
    if guided in ("response_format", "ollama"):
        kwargs["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "narration", "schema": SCHEMA}}
    eb = _extra_body(guided)
    if eb:
        kwargs["extra_body"] = eb
    try:
        cli = cli or _client(url, timeout)
        r = cli.chat.completions.create(**kwargs)
        raw = r.choices[0].message.content or ""
    except Exception as e:  # server down, tunnel expired, bad URL, ...
        return _fallback_result(text, f"{type(e).__name__}: {str(e)[:200]}", t0)
    obj = _to_json(raw)
    ok, sch = _check(obj)
    return {"text": text, "json": obj, "raw": raw, "valid_json": ok, "schema_ok": sch,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "backend": "ollama" if (guided == "ollama" or ":11434" in (url or SLM_URL)) else "vllm", "error": None,
            "completion_tokens": getattr(getattr(r, "usage", None), "completion_tokens", None)}


def parse_batch(lines, url=None, model=None, guided=None, concurrency=16, timeout=30):
    """Parse many lines. Requests are sent *concurrently* so vLLM's continuous batching
    can run them together on the GPU (much higher throughput than one-by-one).

    Returns (results, stats) where stats has wall time, lines/sec and p50/p95 latency.
    """
    lines = [l.strip() for l in lines if l and l.strip()]
    if not lines:
        return [], {"n": 0}
    t0 = time.perf_counter()
    try:
        cli = _client(url, timeout)
    except Exception:
        cli = None  # openai not installed -> parse() falls back per line
    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(lines)))) as ex:
        results = list(ex.map(lambda t: parse(t, url, model, guided, cli, timeout), lines))
    wall = time.perf_counter() - t0
    lat = sorted(r["latency_ms"] for r in results)
    pct = lambda p: lat[min(len(lat) - 1, int(round(p * (len(lat) - 1))))]
    toks = sum(r.get("completion_tokens") or 0 for r in results)
    stats = {"n": len(lines), "wall_s": wall, "lines_per_s": len(lines) / wall if wall else None,
             "p50_ms": pct(0.50), "p95_ms": pct(0.95),
             "output_tokens_per_s": (toks / wall) if toks and wall else None,
             "backends": sorted({r["backend"] for r in results})}
    return results, stats


def health(url=None, timeout=3):
    """True if the server is up. vLLM answers GET /health (200 once the model is loaded);
    Ollama has no /health but answers GET / with "Ollama is running"; llama-server has /health too."""
    import requests
    base = (url or SLM_URL).rstrip("/")
    try:
        if requests.get(base + "/health", timeout=timeout).status_code == 200:
            return True
        r = requests.get(base + "/", timeout=timeout)
        return r.status_code == 200 and "ollama" in r.text.lower()
    except Exception:
        return False


if __name__ == "__main__":
    # Quick CLI test:  SLM_URL=http://localhost:8000 python serving/client.py
    from schema import SAMPLES
    print("server healthy:", health())
    res, st = parse_batch(SAMPLES)
    for r in res:
        print(f"[{r['backend']:14}] {r['latency_ms']:7.1f} ms  schema_ok={r['schema_ok']}  {r['text']}")
        print("   ->", json.dumps(r["json"]))
    print("stats:", json.dumps(st, indent=1))
