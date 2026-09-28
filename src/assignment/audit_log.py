"""
Assignment 11 — Audit log for request and response decisions.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input and start latency timing; return its correlation ID."""
        correlation_id = request_id or str(uuid4())
        self._open[correlation_id] = {
            "request_id": correlation_id,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "_started_monotonic": time.monotonic(),
        }
        return correlation_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, decision layer, and elapsed request time."""
        correlation_id = request_id
        if correlation_id is None:
            correlation_id = next(
                (
                    key
                    for key, entry in reversed(list(self._open.items()))
                    if entry["user_id"] == user_id
                ),
                None,
            )
        pending = self._open.pop(correlation_id, None) if correlation_id else None
        started = pending.get("_started_monotonic") if pending else None
        record = {
            "request_id": correlation_id or str(uuid4()),
            "user_id": user_id,
            "input": pending.get("input", "") if pending else "",
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "timestamp": utc_now_iso(),
            "latency_ms": round((time.monotonic() - started) * 1000, 3)
            if started is not None
            else None,
        }
        if pending:
            record["started_at"] = pending["started_at"]
        self.logs.append(record)
        return record

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
