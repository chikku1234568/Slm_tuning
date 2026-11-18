"""Sanity checks for the CPU + Ollama path. Run from anywhere:

    python serving/cpu_ollama/check_setup.py                 # static checks + live checks if Ollama is up
    SLM_MODEL=finai python serving/cpu_ollama/check_setup.py

1. Modelfile: SYSTEM text == schema.SYS (the training prompt), ChatML stop token, num_ctx/num_predict/temperature.
2. Live (only if Ollama answers on SLM_URL): the prompt Ollama renders has exactly the same token count as the
   hand-written ChatML string (so TEMPLATE is right), client.parse() works in `ollama` mode, and the Streamlit
   app runs end-to-end (streamlit.testing AppTest).
Exit code is non-zero if a check fails. Quality of the answers is NOT checked here: use your eval set for that.
"""
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
sys.path.insert(0, SERVING)
from schema import SYS  # noqa: E402

URL = os.environ.get("SLM_URL", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("SLM_MODEL", "finai")
fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    if not ok:
        fails.append(name)


mf = open(os.path.join(HERE, "Modelfile"), encoding="utf-8").read()
m = re.search(r'^SYSTEM """(.*?)"""', mf, re.S | re.M)
check("Modelfile SYSTEM == schema.SYS", bool(m) and m.group(1) == SYS)
check("Modelfile stop <|im_end|>", 'PARAMETER stop "<|im_end|>"' in mf)
check("Modelfile temperature 0", re.search(r"^PARAMETER temperature 0\s*$", mf, re.M) is not None)
check("Modelfile num_ctx 512 / num_predict 256", "PARAMETER num_ctx 512" in mf and "PARAMETER num_predict 256" in mf)
check("Modelfile TEMPLATE is ChatML", all(t in mf for t in ("<|im_start|>system", "<|im_start|>user", "<|im_start|>assistant")))


def post(path, payload):
    req = urllib.request.Request(URL + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


try:
    up = "ollama" in urllib.request.urlopen(URL + "/", timeout=3).read().decode().lower()
except Exception:
    up = False
if not up:
    print(f"SKIP live checks: Ollama not reachable at {URL}")
else:
    text = "NACH-DR-BAJAJFIN-1234567890"
    # Chat with NO system message: Ollama must inject the Modelfile SYSTEM, so the prompt is exactly the
    # ChatML we expect. Compare token counts with a raw (template-free) request containing that ChatML.
    chat = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 1,
                                         "messages": [{"role": "user", "content": text}]})
    raw = post("/api/generate", {"model": MODEL, "raw": True, "stream": False, "options": {"num_predict": 1},
                                 "prompt": f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"})
    check("rendered prompt token count == expected ChatML", chat["usage"]["prompt_tokens"] == raw["prompt_eval_count"],
          f"chat={chat['usage']['prompt_tokens']} raw={raw['prompt_eval_count']}")

    os.environ.update(SLM_URL=URL, SLM_MODEL=MODEL, SLM_GUIDED="ollama", SLM_CONCURRENCY="1")
    import client  # noqa: E402
    r = client.parse(text, url=URL, model=MODEL, guided="ollama")
    check("client.parse via Ollama (json_schema response_format) -> valid JSON", r["backend"] == "ollama" and r["valid_json"],
          f"{r['latency_ms']:.0f} ms, schema_ok={r['schema_ok']}")

    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(SERVING, "streamlit_app.py"), default_timeout=180).run()
    [b for b in at.button if b.label == "Parse"][0].click().run()
    check("Streamlit app runs end-to-end", not at.exception and len(at.metric) == 5 and not at.warning,
          "metrics=" + str([(x.label, x.value) for x in at.metric]))

print("\nALL OK" if not fails else "\nFAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)
