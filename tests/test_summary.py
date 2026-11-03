import json

from citesure.prompts import FACETS, NOT_FOUND
from citesure.service import (
    CiteSure,
    _summarize_paper_metadata_cached,
    parse_summary_response,
    summarize_paper_metadata,
    summary_word_count,
)
from citesure.store import PaperStore


def test_summary_response_parses_fenced_json_and_all_facets():
    response = "```json\n" + json.dumps(
        {
            "tldr": "A concise overview.",
            "sections": {name: f"Details for {name}." for name in FACETS},
        }
    ) + "\n```"

    tldr, sections = parse_summary_response(response)

    assert tldr == "A concise overview."
    assert list(sections) == list(FACETS)
    assert sections["Method"] == "Details for Method."


def test_summary_word_count_includes_overview_and_sections():
    tldr = "overview " * 25
    sections = {name: "detail " * 25 for name in FACETS}

    assert summary_word_count(tldr, sections) == 150


def test_summary_word_count_covers_both_requested_boundaries():
    sections = {name: "detail " * 25 for name in FACETS}
    assert summary_word_count("overview " * 25, sections) == 150

    sections = {name: "detail " * 34 for name in FACETS}
    assert summary_word_count("overview " * 30, sections) == 200


def test_summary_parser_fills_missing_facets_with_not_found():
    tldr, sections = parse_summary_response('{"tldr":"Overview"}')

    assert tldr == "Overview"
    assert all(value == NOT_FOUND for value in sections.values())


def test_summary_parser_rejects_invalid_json():
    try:
        parse_summary_response("not json")
    except RuntimeError as error:
        assert "invalid summary response" in str(error)
    else:
        raise AssertionError("Invalid JSON should fail explicitly.")


def test_search_many_batches_embedding_and_vector_queries(monkeypatch):
    class FakeVectors(list):
        def tolist(self):
            return list(self)

    class FakeModel:
        def encode(self, queries, **kwargs):
            assert queries == ["Represent this sentence for searching relevant passages: first",
                               "Represent this sentence for searching relevant passages: second"]
            return FakeVectors([[1.0, 0.0], [0.0, 1.0]])

    class FakeCollection:
        def query(self, **kwargs):
            assert kwargs["query_embeddings"] == [[1.0, 0.0], [0.0, 1.0]]
            assert kwargs["n_results"] == 2
            return {
                "ids": [["first-id"], ["second-id"]],
                "documents": [["first text"], ["second text"]],
                "metadatas": [[{"paper_id": "paper"}], [{"paper_id": "paper"}]],
                "distances": [[0.1], [0.2]],
            }

    monkeypatch.setattr("citesure.store.embedder", lambda: FakeModel())
    store = PaperStore.__new__(PaperStore)
    store.col = FakeCollection()

    results = store.search_many(["first", "second"], k=2, paper_ids=["paper"])

    assert [hits[0]["id"] for hits in results] == ["first-id", "second-id"]
    assert [hits[0]["score"] for hits in results] == [0.9, 0.8]


def test_paper_summary_batches_retrieval_and_caches_result(monkeypatch):
    class FakeStore:
        def __init__(self):
            self.search_many_calls = 0

        def has_paper(self, paper_id):
            return paper_id == "paper-id"

        def search_many(self, queries, k, paper_ids):
            self.search_many_calls += 1
            assert queries == list(FACETS.values())
            assert k == 2
            assert paper_ids == ["paper-id"]
            return [
                [{
                    "id": f"chunk-{index}",
                    "text": f"Evidence for {name}.",
                    "meta": {"title": "Test paper", "section": name, "page": index + 1},
                    "score": 0.9,
                }]
                for index, name in enumerate(FACETS)
            ]

    fake_store = FakeStore()
    monkeypatch.setattr("citesure.service.PaperStore", lambda: fake_store)
    calls = []

    def fake_complete(_system, _prompt, max_tokens):
        calls.append(max_tokens)
        return json.dumps({
            "tldr": "A concise overview.",
            "sections": {name: f"Details for {name}." for name in FACETS},
        })

    monkeypatch.setattr("citesure.service.complete", fake_complete)
    service = CiteSure()

    first = service.summarize("paper-id")
    second = service.summarize("paper-id")

    assert first is second
    assert fake_store.search_many_calls == 1
    assert calls == [600]


def test_discovered_paper_summary_caches_result_and_limits_generation_tokens(monkeypatch):
    calls = []

    def fake_complete(_system, _prompt, max_tokens):
        calls.append(max_tokens)
        return json.dumps({
            "tldr": "A concise overview.",
            "sections": {name: f"Details for {name}." for name in FACETS},
        })

    monkeypatch.setattr("citesure.service.complete", fake_complete)
    _summarize_paper_metadata_cached.cache_clear()
    paper = {
        "title": "A Test Research Paper",
        "authors": ["Example Author"],
        "year": 2025,
        "abstract": "An abstract describing the paper's research and findings.",
    }

    first = summarize_paper_metadata(paper)
    second = summarize_paper_metadata(paper)

    assert first == second
    assert calls == [600]
    _summarize_paper_metadata_cached.cache_clear()
