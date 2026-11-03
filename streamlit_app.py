"""CiteSure AI Streamlit experience."""
# ChromaDB needs a newer SQLite on some hosted Streamlit environments.
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import hashlib
import logging
import math
import os
import re
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import streamlit as st

try:
    for _key in (
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENALEX_API_KEY",
        "HF_TOKEN",
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
        if _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    # Local runs may not have a Streamlit secrets file.
    pass

from citesure.config import settings  # noqa: E402
from citesure.ingest import Chunk  # noqa: E402
from citesure.service import CiteSure, summarize_paper_metadata  # noqa: E402
from citesure.store import embedder, is_embedder_loaded  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("citesure.streamlit")

st.set_page_config(
    page_title="CiteSure AI",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --bg: #0b1020;
      --panel: #121a2d;
      --panel2: #18233a;
      --border: #293650;
      --text: #eef3ff;
      --muted: #9baac4;
      --accent: #8b9dff;
      --accent2: #a6f0d0;
    }
    .stApp { background: var(--bg); color: var(--text); }
    [data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
    [data-testid="stHeader"] { background: transparent; }
    h1, h2, h3 { color: var(--text); letter-spacing: -0.025em; }
    p, label, .stCaption { color: var(--muted); }
    div[data-testid="stMetric"] {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 14px; padding: 14px 18px;
    }
    div[data-testid="stMetric"] label { color: var(--muted); }
    .hero {
      background: linear-gradient(120deg, #121a2d, #18233a);
      border: 1px solid var(--border); border-radius: 18px;
      padding: 24px 28px; margin: 4px 0 20px;
    }
    .hero p { margin-bottom: 0; color: var(--muted); }
    .paper-card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 14px; padding: 18px; margin: 10px 0;
    }
    .paper-meta { color: var(--muted); font-size: 0.9rem; }
    .stButton > button[kind="primary"] {
      background: var(--accent); border-color: var(--accent); color: #0b1020;
      font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_app() -> CiteSure:
    return CiteSure()


@st.cache_resource
def prepare_embedding_model():
    return embedder()


pp = get_app()
if not is_embedder_loaded():
    with st.spinner("Loading the paper search model (first startup only)…"):
        prepare_embedding_model()
st.session_state.setdefault("history", [])
st.session_state.setdefault("questions", 0)
st.session_state.setdefault("discover_results", [])
st.session_state.setdefault("discovery_summary", {})
st.session_state.setdefault("paper_summaries", {})


def reconstruct_abstract(index: dict | None) -> str:
    if not index:
        return ""
    words = []
    for word, positions in index.items():
        for position in positions:
            words.append((position, word))
    return " ".join(word for _, word in sorted(words))


def discover_papers(
    query: str,
    exact_year: int | None,
    from_year: int | None,
    to_year: int | None,
    open_access_only: bool,
    sort_by: str,
    limit: int,
) -> list[dict]:
    params = {
        "search.semantic": query,
        "per-page": min(max(limit * 3, 20), 100),
        "select": (
            "id,title,authorships,publication_date,publication_year,"
            "abstract_inverted_index,doi,ids,best_oa_location,"
            "primary_location,open_access,cited_by_count"
        ),
    }
    filters = []
    if exact_year is not None:
        filters.append(f"publication_year:{exact_year}")
    else:
        if from_year is not None:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year is not None:
            filters.append(f"to_publication_date:{to_year}-12-31")
    if open_access_only:
        filters.append("is_oa:true")
    if filters:
        params["filter"] = ",".join(filters)

    api_key = os.getenv("OPENALEX_API_KEY", "").strip()

    try:
        response = httpx.get(
            "https://api.openalex.org/works",
            params=params,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=45.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("OpenAlex returned HTTP %s", exc.response.status_code)
        if exc.response.status_code == 429:
            raise RuntimeError(
                "OpenAlex is rate limiting discovery. Add an OPENALEX_API_KEY "
                "in Streamlit Secrets or try again later."
            ) from exc
        raise RuntimeError(
            f"OpenAlex returned HTTP {exc.response.status_code}. Try again later."
        ) from exc
    except httpx.RequestError as exc:
        logger.exception("Could not connect to OpenAlex")
        raise RuntimeError("Could not connect to OpenAlex. Check the connection and retry.") from exc

    papers = []
    seen = set()
    for work in response.json().get("results") or []:
        authors = []
        for authorship in (work.get("authorships") or [])[:10]:
            author = (authorship.get("author") or {}).get("display_name")
            if author:
                authors.append(author)

        ids = work.get("ids") or {}
        doi = (work.get("doi") or ids.get("doi") or "").replace("https://doi.org/", "")
        title = (work.get("title") or "Untitled paper").strip()
        dedupe_key = doi.lower() or re.sub(r"\W+", "", title.lower())
        if not dedupe_key or dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        year = work.get("publication_year")
        if not isinstance(year, int):
            year = None
        if exact_year is not None and year != exact_year:
            continue
        if from_year is not None and (year is None or year < from_year):
            continue
        if to_year is not None and (year is None or year > to_year):
            continue

        location = work.get("best_oa_location") or {}
        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        open_access = work.get("open_access") or {}
        paper_id = work.get("id") or ids.get("openalex")
        papers.append(
            {
                "id": paper_id,
                "title": title,
                "authors": authors,
                "year": year,
                "publication_date": work.get("publication_date"),
                "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
                "doi": doi,
                "url": f"https://doi.org/{doi}" if doi else None,
                "openalex_url": paper_id,
                "pdf_url": location.get("pdf_url"),
                "is_open_access": bool(open_access.get("is_oa")),
                "citations": int(work.get("cited_by_count") or 0),
                "journal": source.get("display_name"),
            }
        )

    if not papers:
        return []

    model = embedder()
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    paper_texts = [
        f"Title: {paper['title']}\nAbstract: {paper['abstract']}".strip()
        for paper in papers
    ]
    paper_vectors = model.encode(
        paper_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    citation_max = max(
        (math.log1p(max(0, paper["citations"])) for paper in papers),
        default=0.0,
    )
    current_year = date.today().year
    for paper, vector in zip(papers, paper_vectors):
        semantic = float(np.clip(np.dot(vector, query_vector), -1.0, 1.0))
        year = paper["year"]
        recency = 1.0 / (1.0 + max(0, current_year - year) / 10.0) if year else 0.0
        citations = math.log1p(max(0, paper["citations"]))
        citation_score = citations / citation_max if citation_max else 0.0
        paper["semantic_score"] = round(semantic, 4)
        paper["topic_rank_score"] = round(
            0.75 * ((semantic + 1.0) / 2.0)
            + 0.15 * recency
            + 0.10 * citation_score,
            4,
        )

    sort_keys = {
        "Semantic relevance": lambda paper: paper["semantic_score"],
        "Newest first": lambda paper: (paper["year"] or 0, paper["publication_date"] or ""),
        "Most cited": lambda paper: paper["citations"],
        "Smart rank": lambda paper: paper["topic_rank_score"],
    }
    papers.sort(key=sort_keys[sort_by], reverse=True)
    return papers[:limit]


def import_discovered_paper(paper: dict) -> dict:
    title = paper["title"].strip()
    abstract = (paper.get("abstract") or "").strip()
    if len(abstract) < 5:
        raise ValueError("This record has no usable abstract to import.")
    authors = paper.get("authors") or []
    authors_text = ", ".join(authors) if authors else "Unknown"
    year = str(paper.get("year") or "")
    paper_id = hashlib.sha256(
        f"{title}_{paper.get('doi') or ''}_{year}".encode()
    ).hexdigest()[:12]
    if pp.store.has_paper(paper_id):
        return {"paper_id": paper_id, "title": title, "already_exists": True}

    chunk_text = (
        f"Title: {title}\nAuthors: {authors_text}\nYear: {year}\n"
        f"Venue: {paper.get('journal') or 'Academic Publication'}\n\n"
        f"Abstract & Findings:\n{abstract}"
    )
    pp.store.add_chunks(
        [
            Chunk(
                chunk_id=f"{paper_id}_0",
                paper_id=paper_id,
                title=title,
                section="Abstract & Overview",
                page=1,
                text=chunk_text,
            )
        ]
    )
    return {"paper_id": paper_id, "title": title, "already_exists": False}


def show_sources(hits: list[dict], invalid: tuple | list = ()) -> None:
    if invalid:
        st.warning(
            f"The model cited passage(s) {list(invalid)} that were not retrieved. "
            "Please verify those claims manually."
        )
    if not hits:
        st.caption("No supporting passages were retrieved.")
        return
    with st.expander(f"Evidence and citations ({len(hits)} passages)"):
        for index, hit in enumerate(hits, 1):
            meta = hit["meta"]
            st.markdown(
                f"**[{index}] {meta['title']}** · {meta['section']} · "
                f"p. {meta['page']} · relevance {hit['score']:.2f}"
            )
            st.write(hit["text"])
            if index < len(hits):
                st.divider()


def summary_markdown(title: str, result: dict) -> str:
    lines = [f"# {title}", "", "## TL;DR", "", result["tldr"], ""]
    for name, text, _ in result["sections"]:
        lines.extend([f"## {name}", "", text, ""])
    lines.extend(["", f"Word count: {result['word_count']}"])
    return "\n".join(lines)


papers = pp.store.list_papers()
papers.sort(key=lambda paper: paper["title"].casefold())
paper_titles = {paper["paper_id"]: paper["title"] for paper in papers}

with st.sidebar:
    st.markdown("## 📚 Your library")
    st.caption(f"{len(papers)} paper{'s' if len(papers) != 1 else ''} indexed")
    uploaded_files = st.file_uploader(
        "Add one research PDF",
        type=["pdf"],
        accept_multiple_files=False,
        key=f"pdf-upload-{st.session_state.get('upload_widget_version', 0)}",
        help="Upload one PDF at a time. Maximum 25 MB.",
    )
    if uploaded_files:
        uploaded_size = uploaded_files.size
        st.caption(f"{uploaded_size / (1024 * 1024):.1f} MB · uploaded and ready to index")
        if st.button("Index PDF", type="primary", key="index-pdf"):
            if uploaded_size > 25 * 1024 * 1024:
                st.error(f"{uploaded_files.name}: file exceeds the 25 MB limit.")
            elif uploaded_size == 0:
                st.error(f"{uploaded_files.name}: the file is empty.")
            else:
                content = uploaded_files.getbuffer()
                digest = hashlib.sha1(content).hexdigest()[:12]
                safe_name = Path(uploaded_files.name).name
                destination = Path(settings.upload_dir) / f"{digest}_{safe_name}"
                if pp.store.has_paper(digest):
                    st.info(f"{uploaded_files.name} is already in your library.")
                else:
                    try:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("wb") as output:
                            output.write(content)
                        with st.status(
                            "Extracting text and creating search embeddings…",
                            expanded=True,
                        ) as status:
                            added = pp.add_pdf(
                                str(destination)
                            )
                            status.update(label="Paper indexed successfully.", state="complete")
                        st.success(
                            f"Added **{added['title']}** "
                            f"({added['chunks']} passages)."
                        )
                        st.session_state["upload_widget_version"] = (
                            st.session_state.get("upload_widget_version", 0) + 1
                        )
                        st.rerun()
                    except ValueError as exc:
                        destination.unlink(missing_ok=True)
                        st.error(f"{uploaded_files.name}: {exc}")
                    except OSError as exc:
                        logger.exception("Could not save uploaded PDF")
                        destination.unlink(missing_ok=True)
                        st.error(f"Could not save {uploaded_files.name}: {exc}")
                    except Exception as exc:
                        logger.exception("Failed to index uploaded PDF")
                        destination.unlink(missing_ok=True)
                        st.error(f"Could not index {uploaded_files.name}: {exc}")

    library_filter = st.text_input("Filter library", placeholder="Search paper titles")
    visible_papers = [
        paper for paper in papers
        if library_filter.casefold() in paper["title"].casefold()
    ]
    st.caption("Choose papers for Ask and Compare")
    selected_ids = []
    for paper in visible_papers:
        if st.checkbox(
            paper["title"],
            value=True,
            key=f"library-{paper['paper_id']}",
        ):
            selected_ids.append(paper["paper_id"])
    if not papers:
        st.info("Upload a PDF or discover a paper to get started.")

st.markdown(
    """
    <div class="hero">
      <h1>📑 CiteSure AI</h1>
      <p>Trust the evidence behind every citation. Discover literature, explore your papers,
      and get answers grounded in research.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(3)
metric_cols[0].metric("Papers in library", len(papers))
metric_cols[1].metric("Questions this session", st.session_state.questions)
metric_cols[2].metric("Search mode", "Semantic + evidence")

tab_ask, tab_discover, tab_summarize, tab_compare = st.tabs(
    ["💬 Ask", "🔎 Discover", "📝 Summarize", "⚖️ Compare"]
)

with tab_ask:
    st.subheader("Ask your papers")
    st.caption("Answers are grounded in retrieved passages; citations link to the evidence below.")
    if not selected_ids:
        st.info("Select at least one paper in the sidebar, or import a paper from Discover.")
    for role, text in st.session_state.history:
        with st.chat_message(role):
            st.markdown(text)

    question = st.chat_input(
        "Ask about the selected papers…",
        disabled=not selected_ids or st.session_state.questions >= settings.max_questions,
    )
    if question:
        if st.session_state.questions >= settings.max_questions:
            st.error(f"Session question limit ({settings.max_questions}) reached.")
        else:
            previous_history = list(st.session_state.history)
            st.session_state.history.append(("user", question))
            st.session_state.questions += 1
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Searching evidence and generating an answer…"):
                    try:
                        answer = pp.ask(question, selected_ids, history=previous_history)
                    except Exception as exc:
                        logger.exception("Question answering failed")
                        st.error(f"Could not answer the question: {exc}")
                    else:
                        st.markdown(answer["answer"])
                        show_sources(answer["sources"], answer["invalid_citations"])
                        st.session_state.history.append(("assistant", answer["answer"]))

    if st.session_state.history:
        history_text = "\n\n".join(
            f"## {role.title()}\n\n{text}" for role, text in st.session_state.history
        )
        st.download_button(
            "Download Q&A history",
            history_text,
            file_name="citesure-qa-history.md",
            mime="text/markdown",
        )
        if st.button("Clear Q&A history"):
            st.session_state.history = []
            st.session_state.questions = 0
            st.rerun()

with tab_discover:
    st.subheader("Discover research papers")
    st.caption(
        "Search OpenAlex's scholarly works, refine by publication year and access, "
        "then summarize or import papers into your library."
    )
    with st.form("paper-discovery"):
        search_query = st.text_input(
            "Research topic",
            placeholder="e.g. retrieval-augmented generation for scientific literature",
        )
        filter_cols = st.columns(4)
        exact_year = filter_cols[0].number_input(
            "Exact year (0 = any)", min_value=0, max_value=2100, value=0, step=1
        )
        from_year = filter_cols[1].number_input(
            "From year", min_value=0, max_value=2100, value=0, step=1
        )
        to_year = filter_cols[2].number_input(
            "To year", min_value=0, max_value=2100, value=0, step=1
        )
        result_limit = filter_cols[3].selectbox("Results", [5, 10, 20, 30], index=1)
        option_cols = st.columns([1, 2, 1])
        open_access_only = option_cols[0].checkbox("Open access only")
        sort_by = option_cols[1].selectbox(
            "Rank results",
            ["Smart rank", "Semantic relevance", "Newest first", "Most cited"],
        )
        submitted = st.form_submit_button("Search papers", type="primary")

    no_matching_papers = False
    if submitted:
        st.session_state.discover_results = []
        st.session_state.discovery_summary = {}
        if len(search_query.strip()) < 3:
            st.warning("Enter a research topic with at least 3 characters.")
        elif exact_year and (from_year or to_year):
            st.warning("Use an exact year or a year range, not both.")
        elif from_year and to_year and from_year > to_year:
            st.warning("The starting year must be earlier than or equal to the ending year.")
        else:
            with st.spinner("Searching and ranking scholarly papers…"):
                try:
                    st.session_state.discover_results = discover_papers(
                        search_query.strip(),
                        int(exact_year) if exact_year else None,
                        int(from_year) if from_year else None,
                        int(to_year) if to_year else None,
                        open_access_only,
                        sort_by,
                        int(result_limit),
                    )
                    no_matching_papers = not st.session_state.discover_results
                except RuntimeError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    logger.exception("Paper discovery failed")
                    st.error(f"Could not search papers: {exc}")

    results = st.session_state.discover_results
    if results:
        st.success(f"Found {len(results)} papers, ranked by {sort_by.lower()}.")
        for paper in results:
            authors = ", ".join(paper["authors"][:4]) or "Authors not listed"
            meta = " · ".join(
                part for part in (
                    str(paper["year"]) if paper["year"] else None,
                    paper.get("journal"),
                    f"{paper['citations']:,} citations",
                    "Open access" if paper["is_open_access"] else None,
                ) if part
            )
            with st.container(border=True):
                st.markdown(f"### {paper['title']}")
                st.markdown(f"<span class='paper-meta'>{authors} · {meta}</span>", unsafe_allow_html=True)
                st.caption(
                    f"Semantic similarity: {paper['semantic_score']:.2f} · "
                    f"Smart score: {paper['topic_rank_score']:.2f}"
                )
                abstract = paper.get("abstract") or "No abstract was provided by the source."
                st.write(abstract[:1200] + ("…" if len(abstract) > 1200 else ""))
                links = []
                if paper.get("url"):
                    links.append(f"[DOI]({paper['url']})")
                if paper.get("pdf_url"):
                    links.append(f"[Open PDF]({paper['pdf_url']})")
                if paper.get("openalex_url"):
                    links.append(f"[OpenAlex record]({paper['openalex_url']})")
                if links:
                    st.markdown(" · ".join(links))
                action_cols = st.columns([1, 1, 4])
                summary_key = f"discovery-summary-{paper.get('id')}"
                if action_cols[0].button(
                    "Summarize",
                    key=f"summarize-{paper.get('id')}",
                    disabled=not bool(paper.get("abstract")),
                ):
                    with st.spinner("Generating an abstract-based summary…"):
                        try:
                            st.session_state.discovery_summary[summary_key] = (
                                summarize_paper_metadata(paper)
                            )
                        except Exception as exc:
                            logger.exception("Discovered paper summary failed")
                            st.error(f"Could not summarize this paper: {exc}")
                if action_cols[1].button(
                    "Add to library",
                    key=f"import-{paper.get('id')}",
                    disabled=not bool(paper.get("abstract")),
                ):
                    try:
                        imported = import_discovered_paper(paper)
                        message = "Already in library." if imported["already_exists"] else "Added to library."
                        st.success(f"{message} {imported['title']}")
                        st.rerun()
                    except (ValueError, OSError) as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        logger.exception("Could not import discovered paper")
                        st.error(f"Could not import paper: {exc}")
                discovered_summary = st.session_state.discovery_summary.get(summary_key)
                if discovered_summary:
                    with st.expander("Structured summary", expanded=True):
                        st.markdown("**TL;DR**")
                        st.write(discovered_summary["tldr"])
                        for section, text in discovered_summary["sections_dict"].items():
                            st.markdown(f"**{section}**")
                            st.write(text)
                        st.caption(f"{discovered_summary['word_count']} words (approximately)")
                        st.download_button(
                            "Download summary",
                            summary_markdown(paper["title"], {
                                "tldr": discovered_summary["tldr"],
                                "sections": [
                                    (name, text, [])
                                    for name, text in discovered_summary["sections_dict"].items()
                                ],
                                "word_count": discovered_summary["word_count"],
                            }),
                            file_name=f"{re.sub(r'[^A-Za-z0-9-]+', '-', paper['title']).strip('-')[:60] or 'paper'}-summary.md",
                            mime="text/markdown",
                            key=f"download-discovery-{paper.get('id')}",
                        )
    elif no_matching_papers:
        st.info("No matching papers. Try a broader topic or adjust the filters.")

with tab_summarize:
    st.subheader("Summarize a paper")
    if not papers:
        st.info("Upload a PDF or import a result from Discover to generate a summary.")
    else:
        summary_id = st.selectbox(
            "Choose a paper",
            list(paper_titles),
            format_func=lambda paper_id: paper_titles[paper_id],
            key="summary-paper",
        )
        if st.button("Generate structured summary", type="primary"):
            with st.spinner("Retrieving evidence and generating summary…"):
                try:
                    st.session_state.paper_summaries[summary_id] = pp.summarize(summary_id)
                except Exception as exc:
                    logger.exception("Paper summary failed")
                    st.error(f"Could not summarize this paper: {exc}")
        summary = st.session_state.paper_summaries.get(summary_id)
        if summary:
            st.markdown("### TL;DR")
            st.write(summary["tldr"])
            st.progress(min(summary["word_count"] / 250, 1.0), text=f"{summary['word_count']} words · target approximately 250")
            for name, text, _ in summary["sections"]:
                with st.expander(name, expanded=True):
                    st.write(text)
            if summary["sources"]:
                show_sources(summary["sources"])
            st.download_button(
                "Download summary as Markdown",
                summary_markdown(paper_titles[summary_id], summary),
                file_name=f"{re.sub(r'[^A-Za-z0-9-]+', '-', paper_titles[summary_id]).strip('-')[:60] or 'paper'}-summary.md",
                mime="text/markdown",
            )

with tab_compare:
    st.subheader("Compare papers")
    if len(papers) < 2:
        st.info("Add at least two papers to your library to compare them.")
    else:
        compare_ids = st.multiselect(
            "Select 2–5 papers",
            list(paper_titles),
            default=selected_ids[:5],
            format_func=lambda paper_id: paper_titles[paper_id],
            max_selections=5,
        )
        aspect_text = st.text_input(
            "Comparison aspects",
            "problem, method, datasets, results, limitations",
        )
        if st.button(
            "Compare selected papers",
            type="primary",
            disabled=len(compare_ids) < 2,
        ):
            aspects = [aspect.strip() for aspect in aspect_text.split(",") if aspect.strip()]
            if not aspects:
                st.warning("Enter at least one comparison aspect.")
            else:
                with st.spinner("Retrieving evidence and comparing papers…"):
                    try:
                        comparison = pp.compare(compare_ids, aspects)
                        st.session_state.comparison = comparison
                        st.session_state.comparison_ids = compare_ids
                    except Exception as exc:
                        logger.exception("Paper comparison failed")
                        st.error(f"Could not compare papers: {exc}")
        comparison = st.session_state.get("comparison")
        if comparison:
            st.markdown(comparison["answer"])
            show_sources(
                comparison["sources"],
                comparison.get("invalid_citations", ()),
            )
            compare_title = "-".join(
                re.sub(r"[^A-Za-z0-9-]+", "-", paper_titles[paper_id]).strip("-")[:24]
                for paper_id in st.session_state.get("comparison_ids", [])
            )
            st.download_button(
                "Download comparison",
                comparison["answer"],
                file_name=f"citesure-comparison-{compare_title or 'papers'}.md",
                mime="text/markdown",
            )
