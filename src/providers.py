# Ioannis Kalaitzidis, MTE25012

"""
providers.py
============
Αφαίρεση παρόχου μοντέλου (provider layer).

Ένας "provider" μετατρέπει ένα ζεύγος (system_prompt, messages) σε μια απάντηση
και καταγράφει την ωμή κλήση στο transcripts/, ώστε να μπορεί να αναπαραχθεί
αργότερα. Έτσι πετυχαίνουμε τίμια αναπαραγωγιμότητα:

  * "anthropic" / "ollama"  -> πραγματική κλήση μοντέλου (live). Κάθε κλήση
        αποθηκεύεται στο transcripts/<hash>.json με κλειδί (provider, model, system, messages).
  * "replay"                -> δεν καλεί ποτέ δίκτυο. Επιστρέφει το αποθηκευμένο
        transcript μιας κλήσης που έχει ξαναγίνει, αλλιώς πέφτει σε ένα μικρό
        ντετερμινιστικό MOCK ώστε όλη η ροή να τρέχει χωρίς API key και χωρίς
        προηγούμενα transcripts.

Έτσι τα αναφερόμενα νούμερα αναπαράγονται από τα αποθηκευμένα transcripts (replay),
χωρίς να ισχυριζόμαστε ότι το ίδιο το live API είναι ντετερμινιστικό.

Το mock είναι επίτηδες ελάχιστο και ξεκάθαρα σημασμένο. Υπάρχει μόνο για να μπορεί
να τρέξει ο κώδικας offline. Τα πραγματικά αποτελέσματα ασφάλειας προέρχονται από
live provider, όχι από το mock.
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
    """Σταθερό hash που ταυτοποιεί μια κλήση μοντέλου, χρησιμοποιείται ως όνομα του transcript."""
    # Ίδια είσοδος -> ίδιο κλειδί -> ίδιο αποθηκευμένο transcript (βάση του replay).
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
    # Αποθηκεύει ολόκληρη την κλήση και την απάντηση, ώστε να αναπαραχθεί offline.
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
    # Επιστρέφει την αποθηκευμένη απάντηση για ένα κλειδί, ή None αν δεν υπάρχει.
    p = _transcript_path(key)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["response"]
    return None


# ---------------------------------------------------------------------------
# Πρωτόκολλο provider
# ---------------------------------------------------------------------------

class Provider(Protocol):
    # Κάθε provider έχει όνομα, μοντέλο, και μια complete() που δίνει απάντηση.
    name: str
    model: str

    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        ...


# ---------------------------------------------------------------------------
# Ντετερμινιστικό offline mock (εφεδρικό για replay χωρίς transcript)
# ---------------------------------------------------------------------------

def _mock_complete(system: str, messages: List[Dict[str, str]]) -> str:
    """
    Ένα μικροσκοπικό, ντετερμινιστικό υποκατάστατο. ΔΕΝ είναι μοντέλο ασφάλειας,
    απλώς αφήνει τη ροή να τρέξει offline. Παίζει έναν αφελή βοηθό: ακολουθεί τα
    καλόπιστα αιτήματα και συνήθως (όχι πάντα) αρνείται τα προφανή αιτήματα μυστικών,
    ΧΩΡΙΣ ποτέ να εκπέμπει πραγματικό canary token, ώστε οι offline εκτελέσεις να
    είναι ασφαλείς και σταθερές. Η πραγματική συμπεριφορά διαρροής μετριέται με live provider.
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
    """Offline: σερβίρει αποθηκευμένα transcripts, αλλιώς ντετερμινιστικό mock. Χωρίς δίκτυο."""

    name = "replay"

    def __init__(self, model: str = "replay-mock"):
        self.model = model

    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        # Δοκιμάζει τα κλειδιά όλων των live μοντέλων, ώστε το replay να δουλεύει
        # ανεξάρτητα από το ποιος provider παρήγαγε αρχικά το transcript.
        for prov, model in (("anthropic", config.ANTHROPIC_MODEL),
                            ("ollama", config.OLLAMA_MODEL)):
            cached = _load_transcript(_call_key(prov, model, system, messages))
            if cached is not None:
                return cached
        # Δοκιμάζει και transcript αποθηκευμένο κάτω από το ίδιο το replay.
        cached = _load_transcript(_call_key(self.name, self.model, system, messages))
        if cached is not None:
            return cached
        # Τίποτα αποθηκευμένο -> εφεδρικό mock.
        return _mock_complete(system, messages)


class AnthropicProvider:
    """Live Anthropic API. Απαιτεί ANTHROPIC_API_KEY. Αποθηκεύει κάθε κλήση."""

    name = "anthropic"

    def __init__(self, model: Optional[str] = None):
        self.model = model or config.ANTHROPIC_MODEL
        self._client = None

    def _ensure_client(self):
        # Δημιουργεί τον client μόνο όταν χρειαστεί, με σαφή μηνύματα αν λείπει το
        # πακέτο ή το κλειδί.
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
        # Αν υπάρχει αποθηκευμένη η ίδια κλήση, την επιστρέφει χωρίς νέα χρέωση.
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
        # Ενώνει τα κειμενικά μπλοκ της απάντησης σε ένα string.
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        _save_transcript(key, self.name, self.model, system, messages, text)
        return text


class OllamaProvider:
    """Τοπικό μοντέλο ανοικτού κώδικα μέσω Ollama (για το cross-model). Αποθηκεύει κλήσεις."""

    name = "ollama"

    def __init__(self, model: Optional[str] = None):
        self.model = model or config.OLLAMA_MODEL

    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        # Ίδια λογική caching με τους άλλους: αποθηκευμένη κλήση -> άμεση επιστροφή.
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

        # Το Ollama θέλει το system ως πρώτο μήνυμα ρόλου "system".
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
    Επιστρέφει ένα στιγμιότυπο provider. Προεπιλογή: config.PROVIDER.

    Δέχεται προαιρετικό override μοντέλου με σύνταξη "provider:model", π.χ.
    "ollama:mistral" ή "ollama:llama3.1:8b". Αυτό επιτρέπει στο cross-model να
    συγκρίνει δύο διαφορετικά τοπικά μοντέλα χωρίς αλλαγή του config.
    """
    name = (name or config.PROVIDER).lower()

    # Διαχωρισμός προαιρετικού override μοντέλου: "ollama:mistral" -> ("ollama", "mistral")
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
