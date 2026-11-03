# Streamlit Cloud ships an old SQLite that ChromaDB rejects; swap in pysqlite3 when available.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os

import streamlit as st

try:  # Streamlit Cloud: secrets -> environment, before the package reads its settings
    for _k in (
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
        "LLM_MODEL",
        "EMBED_MODEL",
        "CHROMA_DIR",
        "UPLOAD_DIR",
        "TOP_K",
        "CHUNK_WORDS",
        "CHUNK_OVERLAP",
        "MIN_SCORE",
        "USE_HYBRID",
        "USE_RERANK",
        "MAX_QUESTIONS",
    ):
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

from citesure.config import settings  # noqa: E402
from citesure.service import CiteSure  # noqa: E402

st.set_page_config(page_title="CiteSure AI", page_icon="📑", layout="wide")


@st.cache_resource
def get_app() -> CiteSure:
    return CiteSure()


pp = get_app()
st.session_state.setdefault("history", [])
st.session_state.setdefault("indexed", set())
st.session_state.setdefault("questions", 0)


def show_sources(hits, invalid=()):
    if invalid:
        st.warning(f"The model cited passage(s) {list(invalid)} that were not retrieved. Verify manually.")
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        label = f"[{i}] {m['title'][:48]} | {m['section']} | p.{m['page']} | score {h['score']:.2f}"
        with st.expander(label):
            st.write(h["text"])


# ------------------------------- sidebar: library -------------------------------
with st.sidebar:
    st.header("Library")
    for f in st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True) or []:
        if f.name in st.session_state.indexed:
            continue
        with st.spinner(f"Indexing {f.name}..."):
            os.makedirs(settings.upload_dir, exist_ok=True)
            path = os.path.join(settings.upload_dir, os.path.basename(f.name))
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            try:
                pp.add_pdf(path)
                st.session_state.indexed.add(f.name)
            except ValueError as e:
                st.error(f"{f.name}: {e}")
    papers = pp.store.list_papers()
    titles = {p["paper_id"]: p["title"] for p in papers}
    st.caption("Search scope")
    selected = [p["paper_id"] for p in papers
                if st.checkbox(p["title"][:55], value=True, key=p["paper_id"])]

st.title("CiteSure AI")
st.caption("Trust the evidence behind every citation.")
tab_ask, tab_sum, tab_cmp = st.tabs(["Ask", "Summarize", "Compare"])

# ------------------------------------ Ask ---------------------------------------
with tab_ask:
    for role, text in st.session_state.history:
        with st.chat_message(role):
            st.markdown(text)
    if q := st.chat_input("Ask about the selected papers..."):
        if st.session_state.questions >= settings.max_questions:
            st.error(f"Session limit of {settings.max_questions} questions reached. Reload to start over.")
        else:
            st.session_state.questions += 1
            prior = list(st.session_state.history)
            st.session_state.history.append(("user", q))
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Searching and reading..."):
                    try:
                        res = pp.ask(q, selected or None, history=prior)
                    except Exception as e:
                        st.error(f"The language model call failed: {e}")
                        st.stop()
                st.markdown(res["answer"])
                show_sources(res["sources"], res["invalid_citations"])
            st.session_state.history.append(("assistant", res["answer"]))

# ---------------------------------- Summarize -----------------------------------
with tab_sum:
    if not papers:
        st.info("Upload a paper in the sidebar to begin.")
    else:
        pid = st.selectbox("Paper", list(titles), format_func=lambda x: titles[x])
        if st.button("Generate summary"):
            with st.spinner("Reading the paper..."):
                out = pp.summarize(pid)
            st.subheader("TL;DR")
            st.write(out["tldr"])
            for name, text, _ in out["sections"]:
                st.markdown(f"**{name}**")
                st.write(text)

# ----------------------------------- Compare ------------------------------------
with tab_cmp:
    aspects = st.text_input("Aspects (comma-separated)", "problem, method, datasets, results, limitations")
    if len(selected) < 2:
        st.info("Select at least two papers in the sidebar.")
    elif st.button("Compare papers"):
        with st.spinner("Comparing..."):
            res = pp.compare(selected, [a.strip() for a in aspects.split(",") if a.strip()])
        st.markdown(res["answer"])
        show_sources(res["sources"], res["invalid_citations"])
