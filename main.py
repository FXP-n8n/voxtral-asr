import asyncio
import logging
import os

import uvicorn

import config
from api import app, job_store
from model import ModelManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


async def _start_worker(model_mgr: ModelManager):
    await job_store.run_worker(model_mgr.transcribe)


@app.on_event("startup")
async def startup():
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    model_mgr = ModelManager(variant=config.MODEL_VARIANT)
    try:
        model_mgr.load()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise SystemExit(1)

    import api
    api.model_manager = model_mgr

    asyncio.create_task(_start_worker(model_mgr))
    logger.info("Worker started.")


if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
