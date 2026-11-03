import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _flag(key: str) -> bool:
    return os.getenv(key, "false").lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma")
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/papers")
    top_k: int = int(os.getenv("TOP_K", "6"))
    chunk_words: int = int(os.getenv("CHUNK_WORDS", "220"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "40"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.30"))
    use_hybrid: bool = _flag("USE_HYBRID")
    use_rerank: bool = _flag("USE_RERANK")
    max_questions: int = int(os.getenv("MAX_QUESTIONS", "40"))


settings = Settings()
