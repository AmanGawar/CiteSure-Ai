"""System prompts, context formatting and citation validation (Phase 5)."""
import re

NOT_FOUND = "I couldn't find that in the selected papers."

SYSTEM = f"""You are CiteSure AI, a careful research assistant.
Rules:
1. Answer ONLY from the numbered context passages provided.
2. Cite every factual claim with its passage number in square brackets, e.g. [2] or [1][3].
3. If the context does not contain the answer, reply exactly: "{NOT_FOUND}"
4. Never invent numbers, datasets, authors or citations. Copy numbers exactly as written.
5. Be concise: short paragraphs or bullets."""

COMPARE_SYSTEM = SYSTEM + """
Task: compare the papers on the requested aspects. Output a markdown table with one row per
aspect and one column per paper, citing passages like [3] inside cells. Write 'Not stated'
when a paper's passages do not cover an aspect. After the table add 2-3 sentences on the
most important differences."""

FACETS = {
    "Problem and motivation": "What problem does the paper address and why does it matter?",
    "Method": "What is the proposed method or approach? Describe the key components.",
    "Data and experiments": "Which datasets, baselines and evaluation metrics were used?",
    "Key results": "What are the main quantitative results?",
    "Limitations": "What limitations or future work do the authors mention?",
}

SUMMARY_SYSTEM = f"""You are CiteSure AI, an expert research paper summarizer.
Task: Produce an accurate, structured summary of about 250 words total, split into distinct subparts explaining the research paper based ONLY on the provided context passages.

Output your response strictly as valid JSON with this exact structure:
```json
{{
  "tldr": "A 2-3 sentence overarching summary/overview (approx 35 words).",
  "sections": {{
    "Problem and motivation": "What problem does the paper address and why does it matter? Cite sources like [1] (approx 40-50 words).",
    "Method": "The proposed architecture, algorithm, or methodology with key components. Cite sources like [2] (approx 50-60 words).",
    "Data and experiments": "The datasets, benchmarks, baselines, and evaluation metrics used. Cite sources like [3] (approx 40-50 words).",
    "Key results": "The main quantitative findings, metric improvements, and results. Cite sources like [4] (approx 40-50 words).",
    "Limitations": "Limitations, threats to validity, or future work noted by the authors. Cite sources like [5] (approx 30-40 words)."
  }}
}}
```

Rules:
1. Target length is ABOUT 250 WORDS TOTAL across TL;DR and all 5 subparts.
2. Ground all factual statements in the numbered context passages using bracketed citations e.g. [1], [2].
3. If context lacks evidence for a section, write "{NOT_FOUND}" for that section.
4. Output JSON only. Do not add outside commentary."""

METADATA_SUMMARY_SYSTEM = """You are CiteSure AI, an expert scientific research synthesizer.
Task: Given the paper's title, authors, publication year, venue, and abstract, generate an accurate, comprehensive structured summary of ABOUT 250 WORDS total with clear subparts explaining the paper.

Output strictly as valid JSON:
```json
{
  "tldr": "A high-level 2-3 sentence overview of what the paper accomplishes and why it is significant (approx 35-40 words).",
  "sections": {
    "Problem and motivation": "The fundamental research challenge, gap in previous literature, and motivation (approx 40-50 words).",
    "Method": "The core methodology, theoretical framework, architecture, or technique proposed (approx 50-60 words).",
    "Data and experiments": "Datasets, experimental domain, baselines, or evaluation setup (approx 40-50 words).",
    "Key results": "The primary findings, quantitative or qualitative breakthroughs, and outcomes (approx 40-50 words).",
    "Limitations": "Reported limitations, trade-offs, scope constraints, and future research directions (approx 30-40 words)."
  }
}
```
Rules:
1. The total word count across tldr and sections should be approximately 250 words.
2. Be technically accurate, objective, and clear.
3. Output JSON only."""

CITE = re.compile(r"\[(\d+)\]")


def format_context(hits) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h.get("meta") or {}
        title = m.get("title", "Document")
        section = m.get("section", "Section")
        page = m.get("page", 1)
        text = h.get("text", "")
        blocks.append(f"[{i}] ({title} | {section} | p.{page})\n{text}")
    return "\n\n".join(blocks)


def parse_citations(text: str, hits):
    used = sorted({int(n) for n in CITE.findall(text)})
    valid = [n for n in used if 1 <= n <= len(hits)]
    invalid = [n for n in used if n not in valid]
    return valid, invalid
