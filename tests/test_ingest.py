from citesure.ingest import chunk, clean


def test_clean_joins_hyphenated_line_breaks():
    assert clean("learn-\ning is fun") == "learning is fun"


def test_chunks_never_cross_sections_and_respect_budget():
    paras = [("word " * 300, "Introduction", 1), ("word " * 100, "Methods", 2)]
    chunks = chunk(paras, "p1", "Title", target=220, overlap=40)
    assert {c.section for c in chunks} == {"Introduction", "Methods"}
    assert all(len(c.text.split()) <= 300 for c in chunks)


def test_chunk_records_starting_page():
    paras = [("word " * 100, "Methods", 7)]
    assert chunk(paras, "p1", "T")[0].page == 7


def test_zero_overlap_is_respected():
    paras = [("word " * 440, "Methods", 1)]
    chunks = chunk(paras, "p1", "T", target=220, overlap=0)
    assert [len(c.text.split()) for c in chunks] == [220, 220]
