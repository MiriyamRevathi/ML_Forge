"""Persistent local job orchestration for long-running MLForge operations.

The service deliberately keeps execution adapters small and serializable. A job
record is useful even when the process is restarted: callers can inspect the
last known state, progress, messages, retry count, and failure reason.
"""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from config import DATA_DIR
from utils.files import list_files_in_dir, load_json, save_json
from utils.helpers import generate_unique_id


class JobState(str, Enum):
    """States used by the job lifecycle."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = {
    JobState.SUCCEEDED.value,
    JobState.FAILED.value,
    JobState.CANCELLED.value,
}


class JobNotFoundError(LookupError):
    """Raised when a requested job record does not exist."""


class InvalidJobTransition(ValueError):
    """Raised when a caller attempts an invalid lifecycle transition."""


class JobContext:
    """Stateful callbacks exposed to a running job handler."""

    def __init__(self, service: "JobService", job_id: str) -> None:
        self._service = service
        self.job_id = job_id

    def update(self, progress: float, message: str = "") -> Dict[str, Any]:
        """Persist bounded progress and an optional human-readable message."""
        return self._service.update_progress(self.job_id, progress, message)

    def cancellation_requested(self) -> bool:
        """Return whether the caller asked the running handler to stop."""
        return self._service.get(self.job_id)["state"] == JobState.CANCEL_REQUESTED.value

    def checkpoint(self, message: str) -> Dict[str, Any]:
        """Persist a zero-cost progress event without changing the percentage."""
        job = self._service.get(self.job_id)
        return self._service.update_progress(self.job_id, job["progress"], message)


class JobService:
    """Thread-backed job registry with durable JSON records.

    Handlers receive a :class:`JobContext` and may return any JSON-serializable
    value. The executor is intentionally process-local because MLForge's
    default deployment is local and file-backed.
    """

    _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mlforge-job")
    _lock = threading.RLock()
    _futures: Dict[str, Future[Any]] = {}
    _handlers: Dict[str, Callable[[JobContext], Any]] = {}
    _job_dir = DATA_DIR / "jobs"

    @classmethod
    def _now(cls) -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _path(cls, job_id: str) -> Path:
        if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
            raise ValueError("Invalid job identifier")
        return cls._job_dir / f"{job_id}.json"

    @classmethod
    def _write(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        save_json(record, cls._path(record["id"]))
        return record

    @classmethod
    def _transition(cls, record: Dict[str, Any], state: JobState) -> None:
        current = record["state"]
        allowed = {
            JobState.QUEUED.value: {JobState.RUNNING.value, JobState.CANCEL_REQUESTED.value, JobState.CANCELLED.value},
            JobState.RUNNING.value: {JobState.SUCCEEDED.value, JobState.FAILED.value, JobState.CANCEL_REQUESTED.value, JobState.CANCELLED.value},
            JobState.CANCEL_REQUESTED.value: {JobState.CANCELLED.value, JobState.SUCCEEDED.value, JobState.FAILED.value},
            JobState.SUCCEEDED.value: set(),
            JobState.FAILED.value: {JobState.QUEUED.value},
            JobState.CANCELLED.value: {JobState.QUEUED.value},
        }
        if state.value not in allowed.get(current, set()):
            raise InvalidJobTransition(f"Cannot move job from {current} to {state.value}")
        record["state"] = state.value
        record["updated_at"] = cls._now()

    @classmethod
    def create(
        cls,
        job_type: str,
        handler: Callable[[JobContext], Any],
        payload: Optional[Dict[str, Any]] = None,
        max_retries: int = 0,
        auto_start: bool = True,
    ) -> Dict[str, Any]:
        """Create a job, persist it, and optionally submit it immediately."""
        if not job_type or not job_type.strip():
            raise ValueError("job_type is required")
        if not callable(handler):
            raise TypeError("handler must be callable")
        if max_retries < 0 or max_retries > 10:
            raise ValueError("max_retries must be between 0 and 10")
        job_id = generate_unique_id("job")
        now = cls._now()
        record = {
            "id": job_id,
            "type": job_type.strip(),
            "state": JobState.QUEUED.value,
            "payload": payload or {},
            "result": None,
            "error": None,
            "progress": 0.0,
            "message": "Queued",
            "attempt": 0,
            "max_retries": max_retries,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
        }
        with cls._lock:
            cls._write(record)
            cls._handlers[job_id] = handler
            if auto_start:
                cls._submit_locked(job_id)
        return record

    @classmethod
    def _submit_locked(cls, job_id: str) -> None:
        future = cls._executor.submit(cls._run, job_id)
        cls._futures[job_id] = future

    @classmethod
    def start(cls, job_id: str) -> Dict[str, Any]:
        """Start a queued or retryable job."""
        with cls._lock:
            record = cls.get(job_id)
            if record["state"] not in {JobState.QUEUED.value, JobState.FAILED.value, JobState.CANCELLED.value}:
                raise InvalidJobTransition(f"Job {job_id} is not startable")
            if record["state"] in {JobState.FAILED.value, JobState.CANCELLED.value}:
                cls._transition(record, JobState.QUEUED)
                record["error"] = None
                record["message"] = "Queued for retry"
                cls._write(record)
            cls._submit_locked(job_id)
            return record

    @classmethod
    def _run(cls, job_id: str) -> None:
        with cls._lock:
            record = cls.get(job_id)
            if record["state"] == JobState.CANCEL_REQUESTED.value:
                cls._transition(record, JobState.CANCELLED)
                record["finished_at"] = cls._now()
                cls._write(record)
                return
            cls._transition(record, JobState.RUNNING)
            record["attempt"] += 1
            record["started_at"] = record["started_at"] or cls._now()
            record["message"] = "Running"
            cls._write(record)
            handler = cls._handlers.get(job_id)
        if handler is None:
            cls._fail(job_id, "No handler registered for job")
            return
        try:
            result = handler(JobContext(cls, job_id))
            with cls._lock:
                record = cls.get(job_id)
                if record["state"] == JobState.CANCEL_REQUESTED.value:
                    cls._transition(record, JobState.CANCELLED)
                    record["message"] = "Cancelled by caller"
                else:
                    cls._transition(record, JobState.SUCCEEDED)
                    record["progress"] = 100.0
                    record["message"] = "Completed"
                    record["result"] = result
                record["finished_at"] = cls._now()
                cls._write(record)
        except Exception as exc:
            cls._fail(job_id, f"{type(exc).__name__}: {exc}", traceback.format_exc())

    @classmethod
    def _fail(cls, job_id: str, error: str, details: str = "") -> None:
        with cls._lock:
            record = cls.get(job_id)
            if record["state"] not in {JobState.RUNNING.value, JobState.CANCEL_REQUESTED.value}:
                return
            cls._transition(record, JobState.FAILED)
            record["error"] = error
            record["error_details"] = details
            record["message"] = "Failed"
            record["finished_at"] = cls._now()
            cls._write(record)
            if record["attempt"] <= record["max_retries"]:
                cls._transition(record, JobState.QUEUED)
                record["message"] = "Queued for retry"
                cls._write(record)
                cls._submit_locked(job_id)

    @classmethod
    def get(cls, job_id: str) -> Dict[str, Any]:
        """Load a single job record."""
        path = cls._path(job_id)
        if not path.exists():
            raise JobNotFoundError(job_id)
        return load_json(path)

    @classmethod
    def list(cls, states: Optional[Iterable[str]] = None, job_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return newest jobs, optionally filtered by state or type."""
        wanted = set(states or [])
        records = []
        for path in list_files_in_dir(cls._job_dir, extension="json"):
            try:
                record = load_json(path)
            except Exception:
                continue
            if wanted and record.get("state") not in wanted:
                continue
            if job_type and record.get("type") != job_type:
                continue
            records.append(record)
        return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)

    @classmethod
    def update_progress(cls, job_id: str, progress: float, message: str = "") -> Dict[str, Any]:
        """Update progress while a job is running."""
        with cls._lock:
            record = cls.get(job_id)
            if record["state"] not in {JobState.RUNNING.value, JobState.CANCEL_REQUESTED.value}:
                raise InvalidJobTransition("Progress can only be updated for an active job")
            record["progress"] = max(0.0, min(100.0, float(progress)))
            if message:
                record["message"] = str(message)[:500]
            record["updated_at"] = cls._now()
            return cls._write(record)

    @classmethod
    def cancel(cls, job_id: str) -> Dict[str, Any]:
        """Request cancellation; running handlers must honor the context flag."""
        with cls._lock:
            record = cls.get(job_id)
            if record["state"] == JobState.QUEUED.value:
                cls._transition(record, JobState.CANCELLED)
                record["message"] = "Cancelled before execution"
                record["finished_at"] = cls._now()
            elif record["state"] == JobState.RUNNING.value:
                cls._transition(record, JobState.CANCEL_REQUESTED)
                record["message"] = "Cancellation requested"
            else:
                raise InvalidJobTransition(f"Job {job_id} is already terminal")
            return cls._write(record)

    @classmethod
    def purge(cls, before: Optional[datetime] = None, states: Optional[Iterable[str]] = None) -> int:
        """Delete old terminal records and return the number removed."""
        threshold = before or datetime.now(timezone.utc)
        wanted = set(states or TERMINAL_STATES)
        removed = 0
        with cls._lock:
            for record in cls.list(states=wanted):
                stamp = datetime.fromisoformat(record["updated_at"])
                if stamp < threshold:
                    path = cls._path(record["id"])
                    if path.exists():
                        path.unlink()
                        removed += 1
                    cls._handlers.pop(record["id"], None)
                    cls._futures.pop(record["id"], None)
        return removed

    @classmethod
    def shutdown(cls, wait: bool = True) -> None:
        """Stop worker threads, primarily for controlled process shutdown."""
        cls._executor.shutdown(wait=wait, cancel_futures=True)
