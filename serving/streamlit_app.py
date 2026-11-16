"""Streamlit demo: bank statement narration lines -> 10-field JSON via the vLLM server.

    pip install -r serving/requirements-serving.txt
    SLM_URL=https://xxxx.trycloudflare.com streamlit run serving/streamlit_app.py

Streamlit is only a CLIENT: it never loads the model. The model lives in the vLLM
server (Colab GPU / cloud GPU); we talk to it over HTTP. If the server is down the app
falls back to a regex heuristic so you can still see the UI (clearly labelled).
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `import client` works from repo root
import client  # noqa: E402
from schema import SAMPLES  # noqa: E402

st.set_page_config(page_title="Fin AI SLM — Narration → JSON", layout="wide")
st.title("Fin AI SLM — bank statement narration → JSON")

with st.sidebar:
    st.header("Server")
    url = st.text_input("vLLM server URL", os.environ.get("SLM_URL", "http://localhost:8000"))
    model = st.text_input("Served model name", os.environ.get("SLM_MODEL", "finai"))
    guided = st.selectbox("Structured output mode", ["structured_outputs", "guided_json", "response_format", "none"],
                          help="structured_outputs = current vLLM (>=0.12). guided_json = older vLLM. "
                               "none = no constraint (tests the fine-tune alone).")
    concurrency = st.slider("Concurrent requests (batch)", 1, 64, 16,
                            help="Lines are sent in parallel; vLLM batches them on the GPU (continuous batching).")
    up = client.health(url)
    st.markdown("Status: " + (":green[● server up]" if up else ":red[● server unreachable → regex fallback]"))

if "text" not in st.session_state:
    st.session_state.text = SAMPLES[0]
if st.button("Load sample narrations"):
    st.session_state.text = "\n".join(SAMPLES)

text = st.text_area("One narration per line", key="text", height=200)

if st.button("Parse", type="primary"):
    lines = [l for l in text.splitlines() if l.strip()]
    with st.spinner(f"Parsing {len(lines)} line(s)…"):
        results, stats = client.parse_batch(lines, url=url, model=model, guided=guided, concurrency=concurrency)

    if stats.get("n"):
        c = st.columns(5)
        c[0].metric("Lines", stats["n"])
        c[1].metric("Wall time", f"{stats['wall_s']*1000:.0f} ms")
        c[2].metric("Throughput", f"{stats['lines_per_s']:.1f} lines/s")
        c[3].metric("p50 latency", f"{stats['p50_ms']:.0f} ms")
        c[4].metric("p95 latency", f"{stats['p95_ms']:.0f} ms")
        if stats.get("output_tokens_per_s"):
            st.caption(f"Output tokens/s (all requests): {stats['output_tokens_per_s']:.0f}")
        if "regex_fallback" in stats["backends"]:
            st.warning("Some results came from the regex fallback, NOT the model (server unreachable).")

    for r in results:
        with st.container(border=True):
            badges = [
                ":green[valid JSON]" if r["valid_json"] else ":red[invalid JSON]",
                ":green[schema ✓]" if r["schema_ok"] else ":red[schema ✗]",
                f":blue[{r['backend']}]",
                f"{r['latency_ms']:.0f} ms",
            ]
            st.markdown(f"`{r['text']}`  \n" + " · ".join(badges))
            if r["json"] is not None:
                st.json(r["json"], expanded=False)
            else:
                st.code(r["raw"])
            if r["error"]:
                st.caption("error: " + r["error"])
