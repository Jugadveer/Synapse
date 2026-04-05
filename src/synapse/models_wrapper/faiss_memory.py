import numpy as np
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)


class FAISSMemory:
    """
    FAISS-based vector memory with conflict detection and resolution.
    
    Manages semantic memory retrieval with:
    - Entity deduplication
    - Confidence scoring
    - Temporal ordering
    - Dementia-safe soft updates
    """
    
    def __init__(self, dimension=384, memory_dir="faiss_memory"):
        self.dimension = dimension
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # Load embeddings model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS index
        if faiss:
            self.index = faiss.IndexFlatL2(dimension)
        else:
            self.index = None
            logger.warning("FAISS not available; using fallback mode")
        
        # In-memory metadata store
        self.metadata = []  # List of {id, text, embedding, entity, confidence, timestamp}
        self.entity_map = {}  # entity -> list of memory IDs
        
        # Load existing memories
        self._load_memories()
    
    def _load_memories(self):
        """Load existing memories from disk."""
        metadata_file = self.memory_dir / "metadata.json"
        index_file = self.memory_dir / "index.faiss"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)

                # Rebuild entity map
                for record in self.metadata:
                    entity = record.get('entity', 'unknown')
                    if entity not in self.entity_map:
                        self.entity_map[entity] = []
                    self.entity_map[entity].append(record['id'])
            except (json.JSONDecodeError, OSError, TypeError) as e:
                logger.error(f"FAISS metadata load failed, resetting memory store: {e}")
                self.metadata = []
                self.entity_map = {}
                self._quarantine_corrupt_file(metadata_file)
                if index_file.exists() and faiss:
                    self._quarantine_corrupt_file(index_file)
        
        if index_file.exists() and faiss:
            try:
                self.index = faiss.read_index(str(index_file))
            except Exception as e:
                logger.error(f"FAISS index load failed, resetting index: {e}")
                self.index = faiss.IndexFlatL2(self.dimension)
                self._quarantine_corrupt_file(index_file)

    def _quarantine_corrupt_file(self, file_path):
        """Rename a corrupt memory file so the app can recover cleanly."""
        try:
            if not file_path.exists():
                return
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            quarantine_path = file_path.with_suffix(file_path.suffix + f'.bad_{stamp}')
            os.replace(str(file_path), str(quarantine_path))
            logger.info(f"Quarantined corrupt memory file: {quarantine_path}")
        except Exception as e:
            logger.warning(f"Could not quarantine corrupt memory file {file_path}: {e}")
    
    def search(self, query, top_k=3):
        """Search memory by semantic similarity."""
        if not self.metadata:
            return []
        
        query_embedding = self.embedder.encode([query])[0]
        query_tokens = set(re.findall(r"[a-z0-9]+", (query or '').lower()))
        
        if faiss and self.index.ntotal > 0:
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32),
                min(top_k, len(self.metadata))
            )
            
            results = []
            for rank, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.metadata):
                    record = self.metadata[int(idx)]
                    record_tokens = set(re.findall(r"[a-z0-9]+", f"{record.get('text', '')} {record.get('entity', '')}".lower()))
                    lexical_overlap = len(query_tokens & record_tokens)
                    lexical_boost = lexical_overlap * 0.15
                    entity_match = 0.35 if record.get('entity') and record.get('entity').lower() in (query or '').lower() else 0.0
                    base_similarity = 1.0 / (1.0 + float(distances[0][rank]))
                    combined_score = base_similarity + lexical_boost + entity_match
                    results.append({
                        'text': record['text'],
                        'entity': record.get('entity'),
                        'confidence': record.get('confidence', 1.0),
                        'timestamp': record.get('timestamp'),
                        'score': combined_score,
                    })

            results.sort(key=lambda item: item.get('score', 0), reverse=True)
            
            return results
        
        return []
    
    def store(self, text, entity, entity_type, confidence=0.9):
        """
        Store memory with conflict detection.
        
        Args:
            text: Memory content
            entity: Entity name (e.g., "John", "Nurse Sarah")
            entity_type: Type (location|preference|habit|fact|relationship)
            confidence: Confidence score (0-1)
        """
        # Check for conflicts with existing memory of same entity
        conflicts = self._detect_conflicts(entity, entity_type, text)
        
        if conflicts:
            # Handle conflict - soft update for dementia safety
            for conflict_id in conflicts:
                self._resolve_conflict(conflict_id, text, confidence)
        else:
            # New memory
            self._add_memory(text, entity, entity_type, confidence)
        
        self._save_memories()
    
    def _detect_conflicts(self, entity, entity_type, new_text):
        """Find existing memories for same entity."""
        if entity not in self.entity_map:
            return []
        
        conflicts = []
        for memory_id in self.entity_map[entity]:
            record = next((r for r in self.metadata if r['id'] == memory_id), None)
            if record and record.get('entity_type') == entity_type:
                conflicts.append(memory_id)
        
        return conflicts
    
    def _resolve_conflict(self, memory_id, new_text, new_confidence):
        """Resolve memory conflict with dementia-aware logic."""
        record = next((r for r in self.metadata if r['id'] == memory_id), None)
        if not record:
            return
        
        # For dementia: prefer recent, higher confidence updates
        if new_confidence > record.get('confidence', 0):
            record['previous_value'] = record['text']
            record['text'] = new_text
            record['confidence'] = new_confidence
            logger.info(f"Soft update: {record['entity']} -> {new_text}")
    
    def _add_memory(self, text, entity, entity_type, confidence):
        """Add new memory to index."""
        memory_id = len(self.metadata)
        
        # Create embedding
        embedding = self.embedder.encode([text])[0].astype(np.float32)
        
        # Add to FAISS
        if faiss and self.index:
            self.index.add(np.array([embedding]))
        
        # Store metadata
        record = {
            'id': memory_id,
            'text': text,
            'entity': entity,
            'entity_type': entity_type,
            'confidence': confidence,
            'timestamp': np.datetime64('now').item()
        }
        self.metadata.append(record)
        
        # Update entity map
        if entity not in self.entity_map:
            self.entity_map[entity] = []
        self.entity_map[entity].append(memory_id)
        
        logger.info(f"Stored memory: {entity} -> {text}")
    
    def _save_memories(self):
        """Persist memories to disk."""
        metadata_file = self.memory_dir / "metadata.json"
        index_file = self.memory_dir / "index.faiss"
        
        # Convert timestamps to strings for JSON
        serializable = []
        for record in self.metadata:
            r = record.copy()
            if isinstance(r.get('timestamp'), np.datetime64):
                r['timestamp'] = str(r['timestamp'])
            serializable.append(r)
        
        with open(metadata_file, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        if faiss and self.index and self.index.ntotal > 0:
            faiss.write_index(self.index, str(index_file))
