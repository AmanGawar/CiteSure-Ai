
import asyncio
import json
import logging
import os
import re
import shutil
import uuid

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator
from typing import Any, Iterator

import httpx
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from citesure.config import settings
from citesure.service import CiteSure




# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("citesure.api")

app = FastAPI(
    title="CiteSure AI API",
    description=(
        "Research paper assistant with document Q&A, summaries, "
        "comparisons, paper discovery, and embedding-based ranking."
    ),
    version="1.5.0",
)

pp = CiteSure()

MAX_MB = 25
MAX_UPLOAD_BYTES = MAX_MB * 1024 * 1024
OPENALEX_URL = "https://api.openalex.org/works"
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
OPENALEX_RETRY_DELAYS = (1, 2)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5",
)

FRONTEND_PATH = Path(__file__).resolve().parent / "frontend.html"


# ---------------------------------------------------------
# Request models
# ---------------------------------------------------------

class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    paper_ids: list[str] | None = None
    history: list[tuple[str, str]] = Field(default_factory=list)


class CompareIn(BaseModel):
    paper_ids: list[str] = Field(min_length=2, max_length=5)
    aspects: list[str] = Field(
        default_factory=lambda: [
            "problem",
            "method",
            "datasets",
            "results",
            "limitations",
        ]
    )


class SimilarPapersIn(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    year: int | None = Field(default=None, ge=1900, le=2100)
    from_year: int | None = Field(default=None, ge=1900, le=2100)
    to_year: int | None = Field(default=None, ge=1900, le=2100)
    limit: int = Field(default=10, ge=1, le=50)
    open_access_only: bool = Field(default=False)
    sort_by: str = Field(default="smart")  # "smart", "semantic", "recent", "citations"


class SummarizeDiscoveredIn(BaseModel):
    title: str = Field(min_length=1)
    authors: list[str] | str | None = None
    year: int | str | None = None
    journal: str | None = None
    abstract: str | None = None
    doi: str | None = None


class ImportDiscoveredIn(BaseModel):
    title: str = Field(min_length=1)
    abstract: str = Field(min_length=5)
    authors: list[str] | str | None = None
    year: int | str | None = None
    journal: str | None = None
    doi: str | None = None
    pdf_url: str | None = None


# ---------------------------------------------------------
# General helpers
# ---------------------------------------------------------

def safe_filename(filename: str | None) -> str:
    """Remove directory components and unsafe filename characters."""
    name = Path(filename or "paper.pdf").name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    name = name.strip(" .")
    return name or "paper.pdf"


def extract_answer(result: Any) -> str:
    """Normalize common answer return formats."""
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        for key in ("answer", "response", "content", "text"):
            value = result.get(key)
            if isinstance(value, str):
                return value

    return str(result)


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct an OpenAlex abstract from its inverted index."""
    if not isinstance(inverted_index, dict):
        return ""

    positioned_words = []

    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue

        for position in positions:
            if isinstance(position, int) and position >= 0:
                positioned_words.append((position, word))

    positioned_words.sort(key=lambda item: item[0])

    if not positioned_words:
        return ""

    return " ".join(word for _, word in positioned_words)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None

    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.I)

    return doi or None


def get_paper_year(paper: dict) -> int | None:
    year = paper.get("year")

    if year is not None:
        try:
            return int(year)
        except (TypeError, ValueError):
            pass

    publication_date = paper.get("publication_date") or ""

    try:
        return int(publication_date[:4])
    except (TypeError, ValueError):
        return None


def extract_openalex_work(work: dict) -> dict:
    """Convert an OpenAlex record into a frontend-friendly paper object."""
    authorships = work.get("authorships") or []

    authors = []
    for authorship in authorships[:10]:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            authors.append(name)

    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}

    best_location = work.get("best_oa_location") or {}
    open_access = work.get("open_access") or {}
    ids = work.get("ids") or {}

    doi = normalize_doi(work.get("doi") or ids.get("doi"))
    openalex_id = work.get("id") or ids.get("openalex")

    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

    publication_date = work.get("publication_date")
    year = get_paper_year(
        {
            "year": work.get("publication_year"),
            "publication_date": publication_date,
        }
    )

    return {
        "id": openalex_id,
        "title": (work.get("title") or "Untitled paper").strip(),
        "authors": authors,
        "publication_date": publication_date,
        "year": year,
        "abstract": abstract,
        "doi": doi,
        "url": doi and f"https://doi.org/{doi}",
        "openalex_url": openalex_id,
        "pdf_url": best_location.get("pdf_url"),
        "is_open_access": bool(open_access.get("is_oa")),
        "citations": safe_int(work.get("cited_by_count")),
        "journal": source.get("display_name"),
    }


# ---------------------------------------------------------
# Genuine embedding-based semantic similarity & caching
# ---------------------------------------------------------

_QUERY_EMBEDDING_CACHE: dict[str, np.ndarray] = {}
_TEXT_EMBEDDING_CACHE: dict[str, np.ndarray] = {}
_OPENALEX_CACHE: dict[str, tuple[float, dict]] = {}
CACHE_TTL_SECONDS = 1800  # 30 minutes


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Reuse the RAG embedder when discovery uses the same model.
    """
    from citesure.store import embedder

    if settings.embed_model == EMBEDDING_MODEL_NAME:
        logger.info("Using shared embedding model: %s", EMBEDDING_MODEL_NAME)
        return embedder()

    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def paper_embedding_text(paper: dict) -> str:
    """Build text representing a paper for semantic comparison."""
    title = str(paper.get("title") or "").strip()
    abstract = str(paper.get("abstract") or "").strip()

    if title and abstract:
        return f"Title: {title}\nAbstract: {abstract}"

    return title or abstract


def calculate_semantic_scores(
    query: str,
    papers: list[dict],
) -> list[dict]:
    """
    Compute actual cosine similarity between the query embedding
    and each paper's title/abstract embedding, accelerated with in-memory caching.
    """
    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if not papers:
        return papers

    for paper in papers:
        paper["semantic_score"] = 0.0

    model = get_embedding_model()

    # Query embedding (cached)
    if query in _QUERY_EMBEDDING_CACHE:
        query_embedding = _QUERY_EMBEDDING_CACHE[query]
    else:
        query_embedding = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if len(_QUERY_EMBEDDING_CACHE) > 500:
            _QUERY_EMBEDDING_CACHE.clear()
        _QUERY_EMBEDDING_CACHE[query] = query_embedding

    texts = [paper_embedding_text(paper) for paper in papers]

    missing_indices = []
    missing_texts = []
    cached_embeddings: dict[int, np.ndarray] = {}

    for idx, text in enumerate(texts):
        if not text.strip():
            continue
        if text in _TEXT_EMBEDDING_CACHE:
            cached_embeddings[idx] = _TEXT_EMBEDDING_CACHE[text]
        else:
            missing_indices.append(idx)
            missing_texts.append(text)

    if missing_texts:
        new_embeddings = model.encode(
            missing_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for idx, emb, text in zip(missing_indices, new_embeddings, missing_texts):
            if len(_TEXT_EMBEDDING_CACHE) > 2000:
                _TEXT_EMBEDDING_CACHE.clear()
            _TEXT_EMBEDDING_CACHE[text] = emb
            cached_embeddings[idx] = emb

    for idx, paper_emb in cached_embeddings.items():
        similarity = float(np.dot(paper_emb, query_embedding))
        papers[idx]["semantic_score"] = round(
            float(np.clip(similarity, -1.0, 1.0)),
            4,
        )

    return papers


def calculate_paper_scores(
    papers: list[dict],
    query: str,
    current_year: int,
    sort_by: str = "smart",
) -> list[dict]:
    """
    Rank papers using:
      75% semantic similarity
      15% publication recency
      10% citation count
    And sort according to requested sort_by strategy.
    """
    calculate_semantic_scores(query, papers)

    citation_values = [
        max(0, safe_int(paper.get("citations")))
        for paper in papers
    ]

    max_log_citations = max(
        (np.log1p(value) for value in citation_values),
        default=0.0,
    )

    for paper in papers:
        year = get_paper_year(paper)

        if year and year > 0:
            age = max(0, current_year - year)
            recency_score = 1.0 / (1.0 + age / 10.0)
        else:
            recency_score = 0.0

        citations = max(0, safe_int(paper.get("citations")))

        if max_log_citations > 0:
            citation_score = float(
                np.log1p(citations) / max_log_citations
            )
        else:
            citation_score = 0.0

        cosine = float(paper.get("semantic_score") or 0.0)
        semantic_component = (cosine + 1.0) / 2.0

        paper["recency_score"] = round(recency_score, 4)
        paper["citation_score"] = round(citation_score, 4)
        paper["topic_rank_score"] = round(
            0.75 * semantic_component
            + 0.15 * recency_score
            + 0.10 * citation_score,
            4,
        )

    if sort_by == "semantic":
        papers.sort(key=lambda p: float(p.get("semantic_score") or 0.0), reverse=True)
    elif sort_by == "recent":
        papers.sort(
            key=lambda p: (
                get_paper_year(p) or 0,
                str(p.get("publication_date") or ""),
            ),
            reverse=True,
        )
    elif sort_by == "citations":
        papers.sort(key=lambda p: safe_int(p.get("citations")), reverse=True)
    else:  # "smart"
        papers.sort(
            key=lambda paper: paper.get("topic_rank_score", 0.0),
            reverse=True,
        )

    return papers


# ---------------------------------------------------------
# Upload and paper library
# ---------------------------------------------------------

@app.post("/papers")
async def upload_paper(file: UploadFile = File(...)):
    """Upload and index a research paper PDF."""
    original_name = safe_filename(file.filename)

    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Use a unique disk name to prevent collisions between uploads.
    stored_name = f"{uuid.uuid4().hex}_{original_name}"
    path = upload_dir / stored_name

    total_bytes = 0

    try:
        with path.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_MB} MB limit.",
                    )

                output.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        result = await asyncio.to_thread(pp.add_pdf, str(path))
        return result

    except HTTPException:
        path.unlink(missing_ok=True)
        raise

    except ValueError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Paper upload failed")
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Could not process the uploaded paper.",
        ) from exc

    finally:
        await file.close()


@app.get("/papers")
def list_papers():
    """Return the papers already stored in CiteSure."""
    try:
        return pp.store.list_papers()
    except Exception as exc:
        logger.exception("Could not list papers")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve the paper library.",
        ) from exc


@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: str):
    """Delete a paper from the existing paper store."""
    try:
        pp.store.delete_paper(paper_id)
        return {"deleted": paper_id}
    except Exception as exc:
        logger.exception("Could not delete paper %s", paper_id)
        raise HTTPException(
            status_code=500,
            detail="Could not delete the paper.",
        ) from exc


# ---------------------------------------------------------
# Question answering
# ---------------------------------------------------------

@app.post("/ask")
async def ask_papers(body: AskIn):
    """Ask a question about the selected papers."""
    try:
        result = await asyncio.to_thread(
            pp.ask,
            body.question,
            body.paper_ids,
            body.history,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Paper Q&A failed")
        raise HTTPException(
            status_code=500,
            detail="Could not answer the question.",
        ) from exc



@app.post("/ask/stream")
async def ask_papers_stream(body: AskIn):
    """Stream the answer token-by-token using real-time SSE."""
    try:
        from citesure.llm import stream as llm_stream
        from citesure.prompts import SYSTEM, format_context, parse_citations
        from citesure.service import standalone

        query = await asyncio.to_thread(standalone, body.history, body.question)
        hits = await asyncio.to_thread(pp.retrieve, query, body.paper_ids)

        async def event_generator() -> AsyncIterator[str]:
            if not hits:
                sources_payload = json.dumps(
                    {
                        "type": "sources",
                        "sources": [],
                        "invalid_citations": [],
                    },
                    ensure_ascii=False,
                )
                yield f"data: {sources_payload}\n\n"
                not_found_token = json.dumps(
                    {"token": "I couldn't find that in the selected papers."},
                    ensure_ascii=False,
                )
                yield f"data: {not_found_token}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            sources_payload = json.dumps(
                {
                    "type": "sources",
                    "sources": hits,
                    "invalid_citations": [],
                },
                ensure_ascii=False,
            )
            yield f"data: {sources_payload}\n\n"

            user_prompt = f"CONTEXT:\n{format_context(hits)}\n\nQUESTION: {body.question}"
            loop = asyncio.get_running_loop()
            token_queue: asyncio.Queue[str | None] = asyncio.Queue()

            def run_sync_stream():
                try:
                    for token in llm_stream(SYSTEM, user_prompt):
                        loop.call_soon_threadsafe(token_queue.put_nowait, token)
                except Exception as err:
                    logger.exception("Error in LLM stream: %s", err)
                finally:
                    loop.call_soon_threadsafe(token_queue.put_nowait, None)

            stream_task = asyncio.create_task(asyncio.to_thread(run_sync_stream))

            full_text = []
            while True:
                token = await token_queue.get()
                if token is None:
                    break
                full_text.append(token)
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

            await stream_task
            complete_text = "".join(full_text)
            _, invalid = parse_citations(complete_text, hits)
            if invalid:
                invalid_payload = json.dumps(
                    {"type": "invalid_citations", "invalid_citations": invalid},
                    ensure_ascii=False,
                )
                yield f"data: {invalid_payload}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Streaming paper Q&A failed")
        raise HTTPException(
            status_code=500,
            detail="Could not stream the answer.",
        ) from exc



# ---------------------------------------------------------
# Paper summaries
# ---------------------------------------------------------

@app.get("/papers/{paper_id}/summary")
async def summarize_paper(paper_id: str):
    """Generate a summary for a stored paper."""
    try:
        result = await asyncio.to_thread(pp.summarize, paper_id)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Paper summary failed for %s", paper_id)
        raise HTTPException(
            status_code=500,
            detail="Could not summarize the paper.",
        ) from exc


# ---------------------------------------------------------
# Paper comparison
# ---------------------------------------------------------

@app.post("/compare")
async def compare_papers(body: CompareIn):
    """Compare between two and five stored papers."""
    if len(set(body.paper_ids)) != len(body.paper_ids):
        raise HTTPException(
            status_code=422,
            detail="Please select distinct papers for comparison.",
        )

    try:
        result = await asyncio.to_thread(
            pp.compare,
            body.paper_ids,
            body.aspects,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Paper comparison failed")
        raise HTTPException(
            status_code=500,
            detail="Could not compare the selected papers.",
        ) from exc


# ---------------------------------------------------------
# Discover papers using OpenAlex + actual embeddings
# ---------------------------------------------------------

@app.post("/similar-papers")
async def discover_similar_papers(body: SimilarPapersIn):
    """
    Discover papers from OpenAlex and rank them with genuine
    embedding-based cosine similarity.

    OpenAlex finds candidate papers. BGE independently scores each
    candidate's title and abstract against the user's query.
    """
    import time
    query = body.query.strip()

    if len(query) < 3:
        raise HTTPException(
            status_code=422,
            detail="Enter a longer search query.",
        )

    cache_key = (
        f"{query.lower()}_{body.year}_{body.from_year}_{body.to_year}_"
        f"{body.open_access_only}_{body.sort_by}_{body.limit}"
    )
    now = time.time()
    if cache_key in _OPENALEX_CACHE:
        cached_time, cached_payload = _OPENALEX_CACHE[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_payload

    params: dict[str, Any] = {
        "search.semantic": query,
        "per-page": min(max(body.limit * 3, 20), 100),
        "select": (
            "id,title,authorships,publication_date,publication_year,"
            "abstract_inverted_index,doi,ids,best_oa_location,"
            "primary_location,open_access,cited_by_count"
        ),
    }

    filters = []
    if body.year is not None:
        filters.append(f"publication_year:{body.year}")
    else:
        if body.from_year is not None:
            filters.append(f"from_publication_date:{body.from_year}-01-01")
        if body.to_year is not None:
            filters.append(f"to_publication_date:{body.to_year}-12-31")

    if body.open_access_only:
        filters.append("is_oa:true")

    if filters:
        params["filter"] = ",".join(filters)

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            headers = {"Authorization": f"Bearer {OPENALEX_API_KEY}"} if OPENALEX_API_KEY else {}

            for attempt in range(len(OPENALEX_RETRY_DELAYS) + 1):
                response = await client.get(
                    OPENALEX_URL,
                    params=params,
                    headers=headers,
                )

                if (
                    response.status_code == 429
                    and attempt < len(OPENALEX_RETRY_DELAYS)
                ):
                    retry_after = response.headers.get("Retry-After")

                    try:
                        wait_time = max(0, min(float(retry_after), 10))
                    except (TypeError, ValueError):
                        wait_time = OPENALEX_RETRY_DELAYS[attempt]

                    logger.warning(
                        "OpenAlex rate limited. Retrying in %s seconds.",
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                response.raise_for_status()
                payload = response.json()
                break

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        logger.warning("OpenAlex returned HTTP %s", status)
        if status == 429:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Paper discovery is temporarily rate limited. "
                    "Please try again later."
                ),
            ) from exc
        raise HTTPException(
            status_code=502,
            detail="The paper discovery provider returned an error.",
        ) from exc

    except httpx.RequestError as exc:
        logger.exception("Could not connect to OpenAlex")
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the paper discovery provider.",
        ) from exc

    works = payload.get("results") or []

    # Normalize records and remove duplicates.
    papers = []
    seen = set()

    for work in works:
        paper = extract_openalex_work(work)

        doi_key = (paper.get("doi") or "").lower()
        title_key = re.sub(
            r"\W+",
            "",
            (paper.get("title") or "").lower(),
        )

        dedupe_key = doi_key or title_key

        if not dedupe_key or dedupe_key in seen:
            continue

        seen.add(dedupe_key)

        if body.year is not None:
            year = paper.get("year")
            if year is None or year != body.year:
                continue
        else:
            if body.from_year is not None:
                year = paper.get("year")
                if year is None or year < body.from_year:
                    continue
            if body.to_year is not None:
                year = paper.get("year")
                if year is None or year > body.to_year:
                    continue

        if body.open_access_only and not paper.get("is_open_access"):
            continue

        papers.append(paper)

    if not papers:
        empty_res = {
            "query": query,
            "topic": query,
            "count": 0,
            "results": [],
            "papers": [],
            "ranking_method": (
                "75% embedding-based cosine similarity, "
                "15% publication recency, 10% citation count"
            ),
            "ranking_weights": {
                "semantic_similarity": 0.75,
                "recency": 0.15,
                "citations": 0.10,
            },
            "ranking_note": (
                "No matching papers were returned by the discovery provider."
            ),
            "message": "No papers found. Try a broader search query or adjust filters.",
        }
        _OPENALEX_CACHE[cache_key] = (now, empty_res)
        return empty_res

    # Embedding inference is CPU/GPU work, so keep it off the event loop.
    try:
        papers = await asyncio.to_thread(
            calculate_paper_scores,
            papers,
            query,
            date.today().year,
            body.sort_by,
        )
    except Exception as exc:
        logger.exception("Embedding-based ranking failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not calculate semantic similarity. "
                "Check that sentence-transformers and its model are available."
            ),
        ) from exc

    papers = papers[:body.limit]

    res_payload = {
        "query": query,
        "topic": query,
        "count": len(papers),
        "results": papers,
        "papers": papers,
        "ranking_method": (
            "75% embedding-based cosine similarity, "
            "15% publication recency, 10% citation count"
        ),
        "ranking_weights": {
            "semantic_similarity": 0.75,
            "recency": 0.15,
            "citations": 0.10,
        },
        "ranking_note": (
            "Semantic similarity is calculated using normalized "
            "BAAI/bge-small-en-v1.5 embeddings of the search query "
            "and each paper's title and abstract. The semantic_score "
            "is cosine similarity, not a probability. topic_rank_score "
            "combines the rescaled semantic score with recency and citation "
            "signals. OpenAlex supplies the candidate papers."
        ),
        "message": f"Found {len(papers)} papers.",
    }
    _OPENALEX_CACHE[cache_key] = (now, res_payload)
    return res_payload


@app.post("/discover/summarize")
async def summarize_discovered_paper(paper: SummarizeDiscoveredIn):
    """
    Generate an accurate, structured ~250-word summary with subparts
    explaining about the discovered research paper, its authors, year,
    problem, method, data, results, and limitations.
    """
    try:
        from citesure.service import summarize_paper_metadata
        result = await asyncio.to_thread(summarize_paper_metadata, paper.model_dump())
        return result
    except Exception as exc:
        logger.exception("Failed to summarize discovered paper")
        raise HTTPException(
            status_code=500,
            detail="Could not generate summary for this paper.",
        ) from exc


@app.post("/discover/import")
async def import_discovered_paper(body: ImportDiscoveredIn):
    """
    Import a discovered paper's abstract and metadata into the local
    library and ChromaDB index so it can be asked questions,
    summarized, and compared.
    """
    import hashlib
    from citesure.ingest import Chunk

    try:
        title = body.title.strip()
        abstract = body.abstract.strip()
        authors_str = (
            ", ".join(body.authors)
            if isinstance(body.authors, list)
            else str(body.authors or "Unknown")
        )
        year_str = str(body.year or "")

        seed = f"{title}_{body.doi or ''}_{year_str}"
        paper_id = hashlib.sha256(seed.encode()).hexdigest()[:12]

        if pp.store.has_paper(paper_id):
            return {
                "paper_id": paper_id,
                "title": title,
                "already_exists": True,
                "message": "Paper already in your library.",
            }

        chunk_text = (
            f"Title: {title}\n"
            f"Authors: {authors_str}\n"
            f"Year: {year_str}\n"
            f"Venue: {body.journal or 'Academic Publication'}\n\n"
            f"Abstract & Findings:\n{abstract}"
        )

        chunk = Chunk(
            chunk_id=f"{paper_id}_0",
            paper_id=paper_id,
            title=title,
            section="Abstract & Overview",
            page=1,
            text=chunk_text,
        )

        await asyncio.to_thread(pp.store.add_chunks, [chunk])
        return {
            "paper_id": paper_id,
            "title": title,
            "chunks": 1,
            "already_exists": False,
            "message": "Paper successfully imported to library.",
        }
    except Exception as exc:
        logger.exception("Failed to import discovered paper")
        raise HTTPException(
            status_code=500,
            detail="Could not import paper into library.",
        ) from exc


# ---------------------------------------------------------
# Frontend, health, and diagnostics
# ---------------------------------------------------------

@app.get("/health")
def health():
    from citesure.store import is_embedder_loaded

    return {
        "status": "ok",
        "service": "CiteSure AI API",
        "version": app.version,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_model_loaded": (
            get_embedding_model.cache_info().currsize > 0
            or (
                settings.embed_model == EMBEDDING_MODEL_NAME
                and is_embedder_loaded()
            )
        ),
    }


@app.get("/")
def home():
    """Serve the frontend when frontend.html exists beside api.py."""
    if not FRONTEND_PATH.is_file():
        return {
            "service": "CiteSure AI API",
            "version": app.version,
            "message": "API is running. Open /docs for API documentation.",
            "docs": "/docs",
            "health": "/health",
        }

    return FileResponse(
        FRONTEND_PATH,
        media_type="text/html",
    )


@app.get("/favicon.ico")
def favicon():
    raise HTTPException(status_code=404, detail="Favicon not found.")
