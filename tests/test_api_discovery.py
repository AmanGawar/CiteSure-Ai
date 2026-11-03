from fastapi.testclient import TestClient
import api

client = TestClient(api.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "embedding_model" in data


def test_discovery_reuses_rag_embedding_model(monkeypatch):
    model = object()
    monkeypatch.setattr(api, "EMBEDDING_MODEL_NAME", api.settings.embed_model)
    monkeypatch.setattr("citesure.store.embedder", lambda: model)
    api.get_embedding_model.cache_clear()

    try:
        assert api.get_embedding_model() is model
    finally:
        api.get_embedding_model.cache_clear()


def test_discover_summarize_endpoint(monkeypatch):
    def summarize_metadata(_paper_data):
        return {
            "tldr": "A concise overview.",
            "sections_dict": {
                "Problem and motivation": "The research problem.",
                "Method": "The proposed method.",
            },
            "word_count": 150,
        }

    monkeypatch.setattr(
        "citesure.service.summarize_paper_metadata",
        summarize_metadata,
    )
    payload = {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani et al."],
        "year": 2017,
        "journal": "NeurIPS",
        "abstract": (
            "We propose the Transformer, an architecture based solely on attention mechanisms, "
            "dispensing with recurrence and convolutions entirely. Experiments on machine translation "
            "show the model to be superior in quality while being more parallelizable."
        ),
    }
    response = client.post("/discover/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "tldr" in data
    assert "sections_dict" in data
    assert "Problem and motivation" in data["sections_dict"]
    assert "Method" in data["sections_dict"]
    assert data["word_count"] > 100


def test_discover_import_endpoint(monkeypatch):
    monkeypatch.setattr(api.pp.store, "has_paper", lambda _paper_id: False)
    monkeypatch.setattr(api.pp.store, "add_chunks", lambda _chunks: None)
    payload = {
        "title": "Test Paper on Graph Neural Networks",
        "abstract": "Graph Neural Networks (GNNs) capture node and edge features across relational datasets effectively.",
        "authors": ["Smith et al."],
        "year": 2023,
    }
    response = client.post("/discover/import", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "paper_id" in data
    assert data["title"] == "Test Paper on Graph Neural Networks"


def test_similar_papers_retries_rate_limit_with_openalex_key(monkeypatch):
    class FakeAsyncClient:
        calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, params, headers):
            self.calls += 1
            assert headers == {"Authorization": "Bearer test-openalex-key"}
            request = api.httpx.Request("GET", url)
            if self.calls == 1:
                return api.httpx.Response(429, request=request)
            return api.httpx.Response(
                200,
                json={"results": []},
                request=request,
            )

    fake_http_client = FakeAsyncClient()
    monkeypatch.setattr(api, "OPENALEX_API_KEY", "test-openalex-key")
    monkeypatch.setattr(api, "OPENALEX_RETRY_DELAYS", (0,))
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **_kwargs: fake_http_client)
    api._OPENALEX_CACHE.clear()

    response = client.post(
        "/similar-papers",
        json={"query": "language model research", "limit": 10},
    )

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert fake_http_client.calls == 2
