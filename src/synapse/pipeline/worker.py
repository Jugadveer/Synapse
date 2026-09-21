"""Shared worker loop.

Every worker used to wrap its whole `while` in a single try/except, so the
first exception on any turn ended that worker for the rest of the connection.
Nothing restarted it and nothing told the user, so the assistant simply went
quiet. The loop below keeps per-item failures local to the item.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

QUEUE_POLL_TIMEOUT = 0.5


class PipelineWorker:
    """Base class: pulls from one queue and handles items until shutdown."""

    #: Name used in log messages.
    name = 'worker'
    #: Attribute on the pipeline holding this worker's input queue.
    input_queue_name = None

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @property
    def input_queue(self):
        return getattr(self.pipeline, self.input_queue_name)

    async def handle(self, item):  # pragma: no cover - overridden
        raise NotImplementedError

    async def on_error(self, item, exc):
        """Hook for telling the user a turn failed. Default: stay silent."""

    async def run(self):
        while not self.pipeline.shutdown_event.is_set():
            try:
                item = await asyncio.wait_for(
                    self.input_queue.get(), timeout=QUEUE_POLL_TIMEOUT
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if item is None:
                continue

            try:
                await self.handle(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # One bad turn must not take the worker down with it.
                logger.exception(f"[{self.name}] failed to handle item: {exc}")
                try:
                    await self.on_error(item, exc)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(f"[{self.name}] error handler failed too")

        logger.info(f"[{self.name}] shutdown")

    async def aclose(self):
        """Release resources held by the worker. Overridden where needed."""
