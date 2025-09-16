import streamlit as st
import pandas as pd
import time
from EmbeddingContext import LocalEmbeddingContext
from summarize import (
    LLMSettings,
    get_llm_from_settings,
    summarize_single_pass,
    summarize_map_reduce,
)


st.set_page_config(page_title="LightEmbed Demo", layout="wide")


@st.cache_resource(show_spinner=False)
def get_context():
    return {"lec": None}


def build_index(query: str, window: int):
    ctx = get_context()
    # Reuse existing context if present, else create new
    if ctx["lec"] is None:
        ctx["lec"] = LocalEmbeddingContext()
    # Build index for the new query
    start = time.time()
    progress = st.progress(0)
    status = st.empty()

    totals = {"stories": 0, "chunks": 0, "kept": 0}

    def on_progress(p):
        totals.update({k: v for k, v in p.items() if k in totals})
        # We don't have a global total, so animate progress heuristically by chunks processed
        denom = max(1, totals["chunks"])
        pct = min(1.0, totals["kept"] / denom)
        progress.progress(int(pct * 100))
        status.markdown(f"Processed stories: {totals['stories']} • chunks: {totals['chunks']} • kept: {totals['kept']}")

    with st.spinner("Building index (fetching, chunking, embedding, deduping)..."):
        ctx["lec"].build_index_from_query(query, window, progress_callback=on_progress)
    dur = time.time() - start
    progress.progress(100)
    return ctx["lec"], dur


st.title("Featherweight Embeddings: MediaCloud Demo")

with st.sidebar:
    st.header("Build Index")
    mc_query = st.text_area("MediaCloud Query", height=100, value="(police OR victim* OR crim* OR prison OR arrest* OR suspect) AND (\"gun control\" OR \"gun restriction\"~5 OR \"gun law\")")
    mc_window = st.number_input("Days Lookback", min_value=1, max_value=180, value=30, step=1)
    build_btn = st.button("Build / Rebuild Index", type="primary")
    reset_btn = st.button("Reset Context")

ctx = get_context()

if reset_btn:
    ctx["lec"] = None
    st.success("Reset complete. Build a new index to continue.")

if build_btn:
    lec, dur = build_index(mc_query.strip(), int(mc_window))
    st.success(f"Index built in {dur:.1f}s. Kept {lec.count()} chunks.")


st.header("Search")
search_query = st.text_input("Enter search query", value="stories about minnesota")
k = st.slider("Top K", 1, 50, 20)
search_btn = st.button("Run Search")

if search_btn:
    if ctx["lec"] is None or ctx["lec"].index is None:
        st.warning("Please build an index first.")
    else:
        with st.spinner("Searching..."):
            rows = ctx["lec"].search(search_query, k=k)
        if rows is None or rows.empty:
            st.info("No results.")
        else:
            st.dataframe(rows[["score", "publish_date", "title", "text"]], use_container_width=True, height=600)
            st.session_state["last_results"] = rows
            # Update attention chart after search builds context
            try:
                aot = ctx["lec"].attention_over_time()
                if not aot.empty:
                    st.line_chart(aot.set_index("date")[ ["original", "semantic"] ])
            except Exception as e:
                st.info(f"Attention-over-time unavailable: {e}")


st.header("Summarize")
col1, col2 = st.columns(2)
with col1:
    summarize_mode = st.selectbox("Mode", ["single-pass", "map-reduce"], index=0)
    style = st.selectbox("Style", ["bullets", "paragraph"], index=0)
    top_n = st.slider("Summarize Top N from results", 1, 50, min(10, k))
with col2:
    provider = st.selectbox("LLM Provider", ["ollama"], index=0)
    model = st.text_input("Model", value="llama3:8b")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max Output Tokens", 64, 2048, 512, 32)

summarize_btn = st.button("Generate Summary", type="primary")

if summarize_btn:
    rows = st.session_state.get("last_results")
    if rows is None or rows.empty:
        st.warning("Run a search first to get context.")
    else:
        docs = rows.head(top_n).to_dict(orient="records")
        settings = LLMSettings(
            llm_provider=provider,
            llm_model=model,
            llm_temperature=float(temperature),
            llm_max_output_tokens=int(max_tokens),
        )
        output = st.empty()
        output.markdown("Generating summary...")
        try:
            if summarize_mode == "single-pass":
                stream = summarize_single_pass(docs, query=search_query, style=style, settings=settings, stream=True)
            else:
                stream = summarize_map_reduce(docs, query=search_query, settings=settings, stream=True)
            buf = []
            for token in stream:
                buf.append(token)
                # Render incrementally
                output.markdown("".join(buf))
            output.markdown("".join(buf))
        except Exception as e:
            st.error(f"Summarization error: {e}")


