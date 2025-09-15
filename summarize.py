import os
import time
from typing import Iterable, Iterator, List, Dict, Any, Optional, Callable

import requests
import ollama
import tiktoken
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="LLM_", extra="ignore")
    # Provider selection and defaults
    llm_provider: str = "ollama"  # future: "openai", "hf"
    llm_model: str = "llama3:8b"  # default Ollama model tag
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 512
    # Ollama-specific
    ollama_host: str = "http://localhost:11434"


class LLMClient:
    """
    Minimal LLM interface for pluggable providers.
    """
    def name(self) -> str:
        raise NotImplementedError

    def supports_streaming(self) -> bool:
        return False

    def generate(self, prompt: str, stream: bool = False, **params) -> Iterator[str] | str:
        raise NotImplementedError

    def tokenizer_encode(self, text: str) -> List[int]:
        # Default to cl100k_base which is decent for many English prompts
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode(text)


class OllamaLLMClient(LLMClient):
    def __init__(self, model: str, host: str):
        self.model = model
        # Configure client; the official client reads OLLAMA_HOST
        self.host = host.rstrip("/")
        # Keep a dedicated client instance to avoid global env mutation
        self.client = ollama.Client(host=self.host)

    def name(self) -> str:
        return f"ollama:{self.model}"

    def supports_streaming(self) -> bool:
        return True

    def generate(self, prompt: str, stream: bool = False, **params) -> Iterator[str] | str:
        # Map common params to Ollama API: temperature, num_predict (max tokens)
        options = {}
        if "temperature" in params and params["temperature"] is not None:
            options["temperature"] = params["temperature"]
        if "max_tokens" in params and params["max_tokens"] is not None:
            options["num_predict"] = int(params["max_tokens"])  # Ollama naming

        if stream:
            stream_it = self.client.generate(model=self.model, prompt=prompt, stream=True, options=options or None)
            for chunk in stream_it:
                token = chunk.get("response", "")
                if token:
                    yield token
        else:
            res = self.client.generate(model=self.model, prompt=prompt, stream=False, options=options or None)
            return res.get("response", "")


def get_llm_from_settings(settings: Optional[LLMSettings] = None) -> LLMClient:
    s = settings or LLMSettings()
    provider = s.llm_provider.lower()
    if provider == "ollama":
        return OllamaLLMClient(model=s.llm_model, host=s.ollama_host)
    raise ValueError(f"Unsupported llm_provider: {s.llm_provider}")


def build_summary_prompt(query: Optional[str], docs: List[Dict[str, Any]], style: str = "bullets") -> str:
    header = "You are a neutral media analyst. Be concise and specific.\n"
    task = "Summarize the following documents."
    if query:
        task = f"Summarize the following documents with focus on the query: '{query}'."
    if style == "bullets":
        output_req = "Output 5-7 bullet points of key takeaways and a 1-sentence overall summary."
    else:
        output_req = "Output a concise paragraph summary and 3 key takeaways."

    parts = [header, task, output_req, "\nDocuments:"]
    for i, d in enumerate(docs, start=1):
        title = (d.get("title") or "Untitled").strip()
        date = d.get("publish_date") or ""
        text = (d.get("text") or "").strip()
        snippet = text[:1200]
        parts.append(f"[{i}] {title} ({date})\n{snippet}\n")
    parts.append("\nWrite the summary now.")
    return "\n".join(parts)


def summarize_single_pass(
    docs: List[Dict[str, Any]],
    query: Optional[str] = None,
    style: str = "bullets",
    llm: Optional[LLMClient] = None,
    settings: Optional[LLMSettings] = None,
    stream: bool = False,
) -> Iterator[str] | str:
    s = settings or LLMSettings()
    client = llm or get_llm_from_settings(s)
    prompt = build_summary_prompt(query, docs, style=style)
    params = {
        "temperature": s.llm_temperature,
        "max_tokens": s.llm_max_output_tokens,
    }
    return client.generate(prompt, stream=stream, **params)


def map_reduce_chunks(
    docs: List[Dict[str, Any]],
    query: Optional[str],
    llm: LLMClient,
    map_limit_chars: int = 1200,
) -> List[str]:
    notes = []
    for d in docs:
        title = (d.get("title") or "Untitled").strip()
        text = (d.get("text") or "").strip()[:map_limit_chars]
        q = query or ""
        prompt = (
            "You are a concise analyst.\n"
            f"Summarize the following excerpt in <= 80 words with focus on: '{q}'.\n"
            f"Title: {title}\nExcerpt:\n{text}\n\nSummary:"
        )
        out = llm.generate(prompt, stream=False)
        if isinstance(out, str):
            notes.append(out.strip())
        else:
            # If streaming was accidentally enabled, join
            notes.append("".join(out))
    return notes


def build_reduce_prompt(query: Optional[str], notes: List[str]) -> str:
    header = "You are a neutral media analyst."
    task = "Combine these notes into a cohesive, non-redundant summary."
    if query:
        task = f"Combine these notes into a cohesive, non-redundant summary focusing on: '{query}'."
    output_req = "Output 5-7 bullet points and a 1-sentence overall summary."
    body = "\n\n".join(f"- {n.strip()}" for n in notes if n.strip())
    return "\n".join([header, task, output_req, "\nNotes:", body, "\nWrite the summary now."])


def summarize_map_reduce(
    docs: List[Dict[str, Any]],
    query: Optional[str] = None,
    llm: Optional[LLMClient] = None,
    settings: Optional[LLMSettings] = None,
    stream: bool = False,
) -> Iterator[str] | str:
    s = settings or LLMSettings()
    client = llm or get_llm_from_settings(s)
    # Map stage
    notes = map_reduce_chunks(docs, query=query, llm=client)
    # Reduce stage
    reduce_prompt = build_reduce_prompt(query, notes)
    params = {
        "temperature": s.llm_temperature,
        "max_tokens": s.llm_max_output_tokens,
    }
    return client.generate(reduce_prompt, stream=stream, **params)


