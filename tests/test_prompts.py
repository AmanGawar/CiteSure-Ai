from citesure.prompts import format_context, parse_citations

HITS = [{"text": "alpha", "meta": {"title": "T1", "section": "Methods", "page": 3}},
        {"text": "beta", "meta": {"title": "T2", "section": "Results", "page": 5}}]


def test_valid_and_invalid_citations_are_separated():
    valid, invalid = parse_citations("Fact A [1]. Fact B [2][5].", HITS)
    assert valid == [1, 2] and invalid == [5]


def test_context_is_numbered_with_page_info():
    ctx = format_context(HITS)
    assert ctx.startswith("[1] (T1 | Methods | p.3)") and "[2] (T2 | Results | p.5)" in ctx
