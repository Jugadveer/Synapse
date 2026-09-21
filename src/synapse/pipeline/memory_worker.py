import asyncio
import logging

from channels.db import database_sync_to_async

from models_wrapper.faiss_memory import FAISSMemory
from voice.models import MemoryRecord

logger = logging.getLogger(__name__)


class MemoryWorker:
    """Reads and writes the person's memories.

    The embedding model is expensive to load, so the vector store is shared
    across connections. Every read and write is scoped by user_key so one
    person's recollections are never surfaced to another - previously a single
    global store served everybody.
    """

    _shared_faiss = None

    def __init__(self, pipeline):
        self.pipeline = pipeline
        if MemoryWorker._shared_faiss is None:
            MemoryWorker._shared_faiss = FAISSMemory()
        self.faiss = MemoryWorker._shared_faiss

    @property
    def user_key(self):
        return getattr(self.pipeline, 'user_key', None)

    async def run(self):
        """Memory is driven on demand by the router, not by a queue."""
        await self.pipeline.shutdown_event.wait()

    async def aclose(self):
        """The vector store is process-wide; nothing per-connection to free."""

    async def retrieve_context(self, query):
        if not query:
            return ""
        try:
            results = await asyncio.to_thread(
                self.faiss.search, query, 3, self.user_key
            )
            return "\n".join(r['text'] for r in results) if results else ""
        except Exception as e:
            logger.exception(f"Memory retrieval failed: {e}")
            return ""

    async def store_memory(self, entity, entity_type, value):
        """Persist a memory to both the vector store and the database.

        Returns True when the memory was stored.
        """
        value = (value or '').strip()
        if not value:
            return False

        try:
            stored = await asyncio.to_thread(
                self.faiss.store, value, entity, entity_type, 0.9, self.user_key
            )
            if not stored:
                logger.error(f"Vector store rejected memory for {entity!r}")
                return False

            existing = await self._get_memory(entity, entity_type)
            if existing:
                await self._update_memory(existing, value)
                action = 'update'
            else:
                await self._create_memory(entity, entity_type, value)
                action = 'create'

            await self.pipeline.consumer.send_memory_update(action, entity, value)
            return True
        except Exception as e:
            logger.exception(f"Memory storage failed for {entity!r}: {e}")
            return False

    @database_sync_to_async
    def _get_memory(self, entity, entity_type):
        return MemoryRecord.objects.filter(
            entity=entity, entity_type=entity_type, user_key=self.user_key
        ).first()

    @database_sync_to_async
    def _create_memory(self, entity, entity_type, value):
        return MemoryRecord.objects.create(
            entity=entity,
            entity_type=entity_type,
            current_value=value,
            confidence=0.9,
            user_key=self.user_key,
        )

    @database_sync_to_async
    def _update_memory(self, record, new_value):
        record.previous_value = record.current_value
        record.current_value = new_value
        record.save(update_fields=['previous_value', 'current_value', 'updated_at'])
        return record
