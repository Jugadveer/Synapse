import asyncio
import json
import logging
from channels.db import database_sync_to_async
from voice.models import MemoryRecord
from models_wrapper.faiss_memory import FAISSMemory

logger = logging.getLogger(__name__)


class MemoryWorker:
    """Manages FAISS memory retrieval and storage with conflict resolution."""

    _shared_faiss = None
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        if self.__class__._shared_faiss is None:
            self.__class__._shared_faiss = FAISSMemory()
        self.faiss = self.__class__._shared_faiss
    
    async def run(self):
        """Background memory worker (currently passive in intent_queue)."""
        # This worker is called on-demand during routing
        pass
    
    async def retrieve_context(self, query):
        """Retrieve relevant memory for context."""
        try:
            results = await asyncio.to_thread(self.faiss.search, query, top_k=3)
            if results:
                return "\n".join([r['text'] for r in results])
            return ""
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return ""
    
    async def store_memory(self, entity, entity_type, value):
        """Store or update memory with conflict resolution."""
        try:
            # Check for existing memory of same entity
            existing = await self._get_memory(entity, entity_type)
            
            if existing:
                # Handle conflict - prefer latest
                await self._update_memory(existing, value)
                await self.pipeline.consumer.send_memory_update(
                    "update", entity, value
                )
            else:
                # New memory
                await self._create_memory(entity, entity_type, value)
                await self.pipeline.consumer.send_memory_update(
                    "create", entity, value
                )

            # Keep semantic memory in sync with the structured DB record.
            await asyncio.to_thread(self.faiss.store, value, entity, entity_type, 0.9)
        
        except Exception as e:
            logger.error(f"Memory storage error: {e}")
    
    @database_sync_to_async
    def _get_memory(self, entity, entity_type):
        try:
            return MemoryRecord.objects.get(entity=entity, entity_type=entity_type)
        except MemoryRecord.DoesNotExist:
            return None
    
    @database_sync_to_async
    def _create_memory(self, entity, entity_type, value):
        return MemoryRecord.objects.create(
            entity=entity,
            entity_type=entity_type,
            current_value=value,
            confidence=0.9
        )
    
    @database_sync_to_async
    def _update_memory(self, record, new_value):
        record.previous_value = record.current_value
        record.current_value = new_value
        record.save()
