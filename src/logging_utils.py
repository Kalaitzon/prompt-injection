"""
logging_utils.py
================

Each request/response pair is written as one JSON object per line (JSONL) with a
stable schema, a unique interaction_id, and an ISO-8601 timestamp in UTC. This
satisfies the requirement of task 1.

We write to JSONL (not one big JSON array) so that:
  * runs are append-only and crash-resistant,
  * an incomplete run still yields a readable log,
  * sessions can be reconstructed by grouping on session_id.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def new_interaction_id() -> str:
    """Short, unique id for a single request/response interaction."""
    return uuid.uuid4().hex[:12]


def utc_timestamp() -> str:
    """ISO-8601 timestamp in UTC, e.g. 2026-06-16T09:30:00.123456+00:00."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class InteractionRecord:
    """A single logged interaction. Extra details go in `meta` so the schema stays stable."""
    interaction_id: str
    timestamp: str
    session_id: Optional[str]
    turn_index: int
    system: str                # "baseline" or "defended" or a configuration name
    provider: str              # which backend produced the response
    model: str                 # id of the model used
    user_input: str
    assistant_response: str
    task_type: str             # task type (qa, classification, unknown, blocked)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class InteractionLogger:
    """Append-only JSONL logger."""

    def __init__(self, log_file: Path | str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        user_input: str,
        assistant_response: str,
        task_type: str,
        system: str,
        provider: str,
        model: str,
        session_id: Optional[str] = None,
        turn_index: int = 0,
        meta: Optional[Dict[str, Any]] = None,
        interaction_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> InteractionRecord:
        # Builds the record (with auto id/timestamp if not given) and writes it
        # as one line at the end of the file.
        record = InteractionRecord(
            interaction_id=interaction_id or new_interaction_id(),
            timestamp=timestamp or utc_timestamp(),
            session_id=session_id,
            turn_index=turn_index,
            system=system,
            provider=provider,
            model=model,
            user_input=user_input,
            assistant_response=assistant_response,
            task_type=task_type,
            meta=meta or {},
        )
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(record.to_json() + "\n")
        return record

    def read_all(self) -> List[Dict[str, Any]]:
        """Read back all records (for analysis / session reconstruction)."""
        if not self.log_file.exists():
            return []
        records: List[Dict[str, Any]] = []
        with self.log_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


if __name__ == "__main__":
    # Small self-test against a temporary file.
    import tempfile, os

    tmp = Path(tempfile.gettempdir()) / "bank_log_selftest.jsonl"
    if tmp.exists():
        tmp.unlink()
    logger = InteractionLogger(tmp)
    logger.log(
        user_input="What are your opening hours?",
        assistant_response="Branches open 09:00-17:00 on weekdays.",
        task_type="question_answering",
        system="baseline",
        provider="replay",
        model="replay-mock",
        session_id="sess_demo",
        turn_index=0,
    )
    recs = logger.read_all()
    print(f"Wrote and read back {len(recs)} record(s):")
    print(json.dumps(recs[0], indent=2, ensure_ascii=False))
    os.unlink(tmp)
