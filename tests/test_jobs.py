import asyncio
import pytest
from jobs import JobStore, Job, JobStatus


def test_create_job():
    store = JobStore()
    job = store.create("test.mp3")
    assert job.status == JobStatus.PENDING
    assert job.filename == "test.mp3"
    assert job.id in store.jobs


def test_get_job():
    store = JobStore()
    job = store.create("test.mp3")
    fetched = store.get(job.id)
    assert fetched is job


def test_get_missing_job_returns_none():
    store = JobStore()
    assert store.get("nonexistent") is None


def test_job_transitions_to_running():
    store = JobStore()
    job = store.create("test.mp3")
    store.set_running(job.id)
    assert store.get(job.id).status == JobStatus.RUNNING


def test_job_transitions_to_complete():
    from jobs import TranscriptResult, Segment
    store = JobStore()
    job = store.create("test.mp3")
    result = TranscriptResult(
        text="hello world",
        segments=[Segment(start=0.0, end=5.0, text="hello world")]
    )
    store.set_complete(job.id, result)
    j = store.get(job.id)
    assert j.status == JobStatus.COMPLETE
    assert j.result.text == "hello world"


def test_job_transitions_to_error():
    store = JobStore()
    job = store.create("test.mp3")
    store.set_error(job.id, "boom")
    j = store.get(job.id)
    assert j.status == JobStatus.ERROR
    assert j.error == "boom"


def test_queue_size_limit():
    store = JobStore(max_queue=2)
    store.create("a.mp3")
    store.create("b.mp3")
    with pytest.raises(OverflowError):
        store.create("c.mp3")


@pytest.mark.asyncio
async def test_worker_processes_jobs():
    from jobs import TranscriptResult, Segment
    store = JobStore()
    results = []

    async def fake_transcribe(audio_path, language, timestamps):
        results.append(audio_path)
        return TranscriptResult(text="hi", segments=[])

    job = store.create("x.mp3", audio_path="/tmp/x.mp3")
    task = asyncio.create_task(store.run_worker(fake_transcribe))
    await store.queue.join()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert store.get(job.id).status == JobStatus.COMPLETE
    assert "/tmp/x.mp3" in results
