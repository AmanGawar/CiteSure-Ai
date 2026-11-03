"""Retrieval (hit@k, MRR) and answer (faithfulness, correct refusal) evaluation.
Usage: python eval/run_eval.py   (needs a populated index and eval/golden.jsonl)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from citesure.config import settings  # noqa: E402
from citesure.llm import complete  # noqa: E402
from citesure.prompts import NOT_FOUND, SYSTEM, format_context  # noqa: E402
from citesure.store import PaperStore  # noqa: E402

JUDGE = ("You are a strict fact-checker. Decide whether every factual claim in the ANSWER is "
         "supported by the CONTEXT. Reply with JSON only: "
         '{"supported": true, "unsupported_claims": []}')
GOLDEN = Path(__file__).parent / "golden.jsonl"


def load(path=GOLDEN):
    return [json.loads(line) for line in open(path) if line.strip()]


def retrieval_eval(k=None):
    k, store = k or settings.top_k, PaperStore()
    rows = [e for e in load() if e["answerable"]]
    hits_k, rr = 0, 0.0
    for ex in rows:
        hits = store.search(ex["question"], k=k, paper_ids=[ex["paper_id"]])
        rank = next((i for i, h in enumerate(hits, 1)
                     if any(p.lower() in h["text"].lower() for p in ex["evidence"])), None)
        if rank:
            hits_k += 1
            rr += 1 / rank
    res = {"n": len(rows), f"hit@{k}": round(hits_k / max(1, len(rows)), 3), "mrr": round(rr / max(1, len(rows)), 3)}
    print("retrieval", res)
    return res


def answer_eval(limit=30):
    store, faithful, refused_ok, n_ans, n_un = PaperStore(), 0, 0, 0, 0
    for ex in load()[:limit]:
        hits = store.search(ex["question"], k=settings.top_k, paper_ids=[ex["paper_id"]])
        ctx = format_context(hits)
        out = complete(SYSTEM, f"CONTEXT:\n{ctx}\n\nQUESTION: {ex['question']}")
        if ex["answerable"]:
            n_ans += 1
            raw = complete(JUDGE, f"CONTEXT:\n{ctx}\n\nANSWER:\n{out}", max_tokens=400)
            try:
                verdict = json.loads(raw.strip().strip("`").removeprefix("json"))
                faithful += bool(verdict["supported"])
            except Exception:
                pass                            # unparseable judge output counts as a miss
        else:
            n_un += 1
            refused_ok += NOT_FOUND in out
    res = {"faithful": f"{faithful}/{n_ans}", "correct_not_found": f"{refused_ok}/{n_un}"}
    print("answers", res)
    return res


if __name__ == "__main__":
    out = {"retrieval": retrieval_eval(), "answers": answer_eval()}
    (Path(__file__).parent / "results.json").write_text(json.dumps(out, indent=2))
