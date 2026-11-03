"""The ONLY file that talks to the LLM. Supports Groq (fast inference) and Anthropic."""
import os
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings

_groq_client = None
_anthropic_client = None


def get_client():
    global _groq_client, _anthropic_client
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and groq_key.strip():
        if _groq_client is None:
            import groq
            _groq_client = groq.Groq(api_key=groq_key.strip())
        return "groq", _groq_client

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and anthropic_key.strip():
        if _anthropic_client is None:
            import anthropic
            _anthropic_client = anthropic.Anthropic(api_key=anthropic_key.strip())
        return "anthropic", _anthropic_client

    raise ValueError("Neither GROQ_API_KEY nor ANTHROPIC_API_KEY is set in environment.")


def _calc_max_tokens(requested: int) -> int:
    # If using reasoning models like openai/gpt-oss-*, internal reasoning consumes tokens
    if "oss" in settings.llm_model.lower():
        return max(requested, 3500)
    return requested


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
def complete(system: str, user: str, max_tokens: int = 1500) -> str:
    provider, client = get_client()
    tokens = _calc_max_tokens(max_tokens)
    if provider == "groq":
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=tokens,
            temperature=0.1,
        )
        return (response.choices[0].message.content or "").strip()
    else:
        msg = client.messages.create(
            model=settings.llm_model,
            max_tokens=tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()


def stream(system: str, user: str, max_tokens: int = 1500):
    provider, client = get_client()
    tokens = _calc_max_tokens(max_tokens)
    if provider == "groq":
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=tokens,
            temperature=0.1,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
    else:
        with client.messages.stream(
            model=settings.llm_model,
            max_tokens=tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as s:
            yield from s.text_stream
