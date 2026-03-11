import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    text: str
    segments: list[Segment]


@dataclass
class Job:
    id: str
    filename: str
    status: JobStatus
    audio_path: Optional[str]
    language: Optional[str]
    timestamps: bool
    created_at: datetime
    result: Optional[TranscriptResult] = None
    error: Optional[str] = None


class JobStore:
    def __init__(self, max_queue: int = 10):
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self._max_queue = max_queue

    def _pending_count(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status == JobStatus.PENDING)

    def create(
        self,
        filename: str,
        audio_path: Optional[str] = None,
        language: Optional[str] = None,
        timestamps: bool = False,
    ) -> Job:
        if self._pending_count() >= self._max_queue:
            raise OverflowError("Queue full")
        job = Job(
            id=str(uuid.uuid4()),
            filename=filename,
            status=JobStatus.PENDING,
            audio_path=audio_path,
            language=language,
            timestamps=timestamps,
            created_at=datetime.now(timezone.utc),
        )
        self.jobs[job.id] = job
        self.queue.put_nowait(job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def set_running(self, job_id: str) -> None:
        self.jobs[job_id].status = JobStatus.RUNNING

    def set_complete(self, job_id: str, result: TranscriptResult) -> None:
        job = self.jobs[job_id]
        job.status = JobStatus.COMPLETE
        job.result = result

    def set_error(self, job_id: str, error: str) -> None:
        job = self.jobs[job_id]
        job.status = JobStatus.ERROR
        job.error = error

    async def run_worker(self, transcribe_fn) -> None:
        while True:
            job_id = await self.queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                self.queue.task_done()
                continue
            self.set_running(job_id)
            try:
                result = await transcribe_fn(
                    job.audio_path, job.language, job.timestamps
                )
                self.set_complete(job_id, result)
            except Exception as exc:
                self.set_error(job_id, str(exc))
            finally:
                self.queue.task_done()
