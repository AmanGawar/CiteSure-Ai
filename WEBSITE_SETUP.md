# CiteSure AI standalone website

The project serves a modern, responsive research assistant website from the FastAPI app at `/`, powered by a high-speed Retrieval-Augmented Generation (RAG) backend and semantic search.

## Run locally (Windows PowerShell)

1. Open the project folder `citesure-main` in VS Code / PowerShell.
2. Activate your virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Copy `.env.example` to `.env` and add your Groq API key (or an Anthropic API key):

   ```env
   GROQ_API_KEY=your_key_here
   ```

   The default model is `qwen/qwen3.8-27b`; set `LLM_MODEL` in `.env` to override it.

4. Start the website and API:

   ```powershell
   uvicorn api:app --reload
   ```

5. Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Interactive API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## Key Capabilities & Features

### 1. Literature Discovery with Year Filters & Semantic Ranking
- **Topic Search**: Enter any research question, methodology, or topic. CiteSure searches OpenAlex (250M+ scholarly works) and ranks results using local `BAAI/bge-small-en-v1.5` embeddings.
- **Publication Year Filters**:
  - Filter by exact publication year (e.g. `2024`, `2023`, etc.).
  - Filter by year range (e.g. from year `2022` onwards).
- **Ranking & Sorting Options**:
  - **Smart Hybrid Rank**: 75% semantic similarity + 15% recency + 10% citation weight.
  - **Semantic Relevance**: Pure cosine similarity.
  - **Publication Date**: Newest papers first.
  - **Citation Count**: Most cited papers first.
- **Open Access Filter**: Toggle to show only open-access papers with free PDF/full-text access.
- **AI Summary (~250 words)**: Generate a structured analysis of a discovered paper using the metadata and abstract returned by OpenAlex.
- **📥 Add to Library & Ask**: Instantly import any discovered paper into your local library and jump straight into evidence-based Q&A without needing a local file.

### 2. Fast RAG & ~250-Word Structured Summaries
- **Ultra-Fast Generation**: Replaces multi-turn sequential queries with single-pass structured extraction on Groq's high-speed inference engine, cutting generation time from 15+ seconds down to ~1 second.
- **Subparts Explaining the Research Paper**:
  - 📌 **Executive Overview / TL;DR**: 2–3 sentence high-level overview.
  - 🎯 **Problem and Motivation**: The core research challenge and rationale.
  - ⚙️ **Proposed Methodology**: Architecture, algorithms, and key components.
  - 📊 **Datasets & Experiments**: Datasets, benchmarks, and evaluation metrics.
  - 🏆 **Key Results & Findings**: Quantitative outcomes and breakthroughs.
  - ⚠️ **Limitations & Future Work**: Constraints, trade-offs, and future work.
- **Word Count Verification**: Includes live word-count verification badge targeting ~250 words.
- **Grounded Evidence**: Every statement is tied to specific passage citations `[1]`, `[2]` with section and page numbers.

### 3. Accurate Evidence-Based Q&A (Ask Papers)
- **Dense Vector Search**: Powered by BGE-small cosine similarity search in ChromaDB.
- **Grounded Responses**: Answers strictly from retrieved text; refuses unknown answers to prevent hallucinations.
- **Real-Time Token Streaming**: Server-Sent Events (SSE) stream tokens directly from Groq so answers begin rendering in milliseconds.
- **Citation Verification**: Programmatically audits bracketed references `[n]` against retrieved evidence.

---

## Running Unit Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```
