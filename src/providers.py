"""
providers.py
============
Model provider abstraction (provider layer).

A "provider" turns a (system_prompt, messages) pair into a response and records
the raw call in transcripts/, so that it can be replayed later. This is how we
achieve honest reproducibility:

  * "anthropic" / "ollama"  -> a real (live) model call. Each call is stored in
        transcripts/<hash>.json keyed on (provider, model, system, messages).
  * "replay"                -> never calls the network. Returns the stored
        transcript of a previously-made call, otherwise falls back to a small
        deterministic MOCK so that the whole flow runs without an API key and
        without prior transcripts.

This way the reported numbers are reproduced from the stored transcripts (replay),
without claiming that the live API itself is deterministic.

The mock is deliberately minimal and clearly marked. It exists only so the code
can run offline. The actual security results come from a live provider, not from
the mock.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import config


# ---------------------------------------------------------------------------
# Message type
# ---------------------------------------------------------------------------
# messages: list of {"role": "user"|"assistant", "content": str}


def _call_key(provider: str, model: str, system: str, messages: List[Dict[str, str]]) -> str:
    """Stable hash identifying a model call, used as the transcript filename."""
    # Same input -> same key -> same stored transcript (the basis of replay).
    payload = json.dumps(
        {"provider": provider, "model": model, "system": system, "messages": messages},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _transcript_path(key: str) -> Path:
    return config.TRANSCRIPTS_DIR / f"{key}.json"


def _save_transcript(key: str, provider: str, model: str, system: str,
                     messages: List[Dict[str, str]], response: str) -> None:
    # Stores the entire call and response, so it can be replayed offline.
    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    _transcript_path(key).write_text(
        json.dumps(
            {
                "key": key,
                "provider": provider,
                "model": model,
                "system": system,
                "messages": messages,
                "response": response,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_transcript(key: str) -> Optional[str]:
    # Returns the stored response for a key, or None if it does not exist.
    p = _transcript_path(key)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["response"]
    return None


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

class Provider(Protocol):
    # Each provider has a name, a model, and a complete() that returns a response.
    name: str
    model: str

    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        ...


# ---------------------------------------------------------------------------
# Deterministic offline mock (fallback for replay without a transcript)
# ---------------------------------------------------------------------------

def _mock_complete(system: str, messages: List[Dict[str, str]]) -> str:
    """
    A tiny, deterministic stand-in. It is NOT a security model; it merely lets the
    flow run offline. It plays a naive assistant: it follows benign requests and
    usually (not always) refuses the obvious secret requests, WITHOUT ever emitting
    a real canary token, so that offline runs are safe and stable. The actual leak
    behavior is measured with a live provider.
    """
    last_user = ""
    for m in reversed(messages):
        if m["role"] == "user":
            last_user = m["content"]
            break
    text = last_user.lower()

    benign_kw = ("summar", "opening hours", "how do i", "what is", "classify",
                 "categor", "branch", "card", "balance", "fee")
    if any(k in text for k in benign_kw):
        return ("[MOCK] Thanks for contacting Bank Support. "
                "Here is a general, non-confidential answer to your question.")
    # default: cautious refusal, no secret content
    return ("[MOCK] I can help with general Bank questions, but I can't share "
            "internal or confidential information.")


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------

class ReplayProvider:
    """Offline: serves stored transcripts, otherwise a deterministic mock. No network."""

    name = "replay"

    def __init__(self, model: str = "replay-mock"):
        self.model = model

    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        # Try the keys of all live models, so that replay works regardless of
        # which provider originally produced the transcript.
        for prov, model in (("anthropic", config.ANTHROPIC_MODEL),
                            ("ollama", config.OLLAMA_MODEL)):
            cached = _load_transcript(_call_key(prov, model, system, messages))
            if cached is not None:
                return cached
        # Also try a transcript stored under replay itself.
        cached = _load_transcript(_call_key(self.name, self.model, system, messages))
        if cached is not None:
            return cached
        # Nothing stored -> fallback mock.
        return _mock_complete(system, messages)


class AnthropicProvider:
    """Live Anthropic API. Requires ANTHROPIC_API_KEY. Stores every call."""

    name = "anthropic"

    def __init__(self, model: Optional[str] = None):
        self.model = model or config.ANTHROPIC_MODEL
        self._client = None

    def _ensure_client(self):
        # Creates the client only when needed, with clear messages if the package
        # or the key is missing.
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise RuntimeError(
                    "The 'anthropic' package is not installed. Run "
                    "`pip install anthropic`, or use BANK_PROVIDER=replay."
                ) from e
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Set it in your environment "
                    "(.env), or use BANK_PROVIDER=replay for offline runs."
                )
            self._client = Anthropic()

    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        # If the same call is already stored, return it without a new charge.
        key = _call_key(self.name, self.model, system, messages)
        cached = _load_transcript(key)
        if cached is not None:
            return cached

        self._ensure_client()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            system=system,
            messages=messages,
        )
        # Joins the text blocks of the response into a single string.
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        _save_transcript(key, self.name, self.model, system, messages, text)
        return text


class OllamaProvider:
    """Local open-source model via Ollama (for the cross-model study). Stores calls."""

    name = "ollama"

    def __init__(self, model: Optional[str] = None):
        self.model = model or config.OLLAMA_MODEL

    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        # Same caching logic as the others: stored call -> immediate return.
        key = _call_key(self.name, self.model, system, messages)
        cached = _load_transcript(key)
        if cached is not None:
            return cached

        try:
            import ollama  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "The 'ollama' package is not installed. Run `pip install ollama` "
                "and ensure the Ollama server is running, or use BANK_PROVIDER=replay."
            ) from e

        # Ollama wants the system as the first message with role "system".
        chat_messages = [{"role": "system", "content": system}] + messages
        resp = ollama.chat(
            model=self.model,
            messages=chat_messages,
            options={"temperature": config.TEMPERATURE},
        )
        text = resp["message"]["content"]
        _save_transcript(key, self.name, self.model, system, messages, text)
        return text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(name: Optional[str] = None) -> Provider:
    """
    Returns a provider instance. Default: config.PROVIDER.

    Accepts an optional model override with the syntax "provider:model", e.g.
    "ollama:mistral" or "ollama:llama3.1:8b". This lets the cross-model study
    compare two different local models without changing the config.
    """
    name = (name or config.PROVIDER).lower()

    # Split the optional model override: "ollama:mistral" -> ("ollama", "mistral")
    model_override: Optional[str] = None
    if ":" in name:
        first, rest = name.split(":", 1)
        if first in ("ollama", "anthropic", "replay"):
            name, model_override = first, rest

    if name == "replay":
        return ReplayProvider()
    if name == "anthropic":
        return AnthropicProvider(model=model_override)
    if name == "ollama":
        return OllamaProvider(model=model_override)
    raise ValueError(f"Unknown provider: {name!r}")


if __name__ == "__main__":
    p = get_provider()
    print(f"Provider: {p.name} (model={p.model})")
    demo_msgs = [{"role": "user", "content": "What are your opening hours?"}]
    print("Response:", p.complete("You are Bank Support.", demo_msgs))
