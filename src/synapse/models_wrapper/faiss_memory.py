import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - depends on the install
    SentenceTransformer = None

try:
    import faiss
except ImportError:  # pragma: no cover - depends on the install
    faiss = None

logger = logging.getLogger(__name__)

# Anchor the store to the package rather than the working directory, so the
# memory a person builds up does not depend on where the server was launched.
DEFAULT_MEMORY_DIR = Path(__file__).resolve().parents[1] / "faiss_memory"


class FAISSMemory:
    """
    FAISS-backed semantic memory with conflict detection and resolution.

    Persistence model: metadata.json and embeddings.npy are the source of
    truth and are always written together. The FAISS index is a derived cache
    rebuilt from the embeddings, so the index can never drift out of step with
    the records it points at.
    """

    def __init__(self, dimension=384, memory_dir=None):
        self.dimension = dimension
        self.memory_dir = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Without the embedding model semantic recall is unavailable, but
        # reminders and conversation still work, so degrade instead of
        # taking the whole websocket down at connect time.
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') if SentenceTransformer else None
        if self.embedder is None:
            logger.error(
                "sentence-transformers is not installed; semantic memory is disabled. "
                "Install the requirements to enable recall."
            )

        self.metadata = []          # list of record dicts
        self.embeddings = None      # (n, dimension) float32, row i <-> metadata[i]
        self.entity_map = {}        # entity -> list of positions in metadata
        self.index = None

        self._load_memories()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    @property
    def _metadata_file(self):
        return self.memory_dir / "metadata.json"

    @property
    def _embeddings_file(self):
        return self.memory_dir / "embeddings.npy"

    def _load_memories(self):
        """Load records and embeddings, then rebuild the index from them."""
        metadata = []
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                if not isinstance(metadata, list):
                    raise TypeError(f"expected a list of records, got {type(metadata).__name__}")
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
                logger.error(f"Memory metadata load failed, resetting store: {e}")
                self._quarantine_corrupt_file(self._metadata_file)
                self._quarantine_corrupt_file(self._embeddings_file)
                metadata = []

        embeddings = None
        if metadata and self._embeddings_file.exists():
            try:
                embeddings = np.load(self._embeddings_file).astype(np.float32)
            except (OSError, ValueError) as e:
                logger.error(f"Memory embeddings load failed, will re-embed: {e}")
                self._quarantine_corrupt_file(self._embeddings_file)
                embeddings = None

        # Re-embed rather than drop the records if the vectors are missing or
        # do not line up with the metadata; the text is what we cannot recreate.
        if metadata and (embeddings is None or embeddings.shape[0] != len(metadata)):
            if embeddings is not None:
                logger.warning(
                    f"Embedding count {embeddings.shape[0]} != record count {len(metadata)}; re-embedding"
                )
            embeddings = self._encode([r.get('text', '') for r in metadata])

        self.metadata = metadata
        self.embeddings = embeddings if metadata else None
        self._rebuild_entity_map()
        self._rebuild_index()

        if self.metadata:
            logger.info(f"Loaded {len(self.metadata)} memories from {self.memory_dir}")

    def _save_memories(self):
        """Persist records and embeddings atomically.

        Written to temporary files first and swapped into place, so an
        interrupted or failing write can never leave a half-written store
        behind for the next startup to choke on.
        """
        try:
            self._write_atomic(
                self._metadata_file,
                lambda path: self._dump_json(path, self.metadata),
            )
            if self.embeddings is not None and len(self.metadata):
                self._write_atomic(
                    self._embeddings_file,
                    lambda path: np.save(str(path), self.embeddings, allow_pickle=False),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to persist memory store: {e}")
            return False

    @staticmethod
    def _dump_json(path, payload):
        with open(path, 'w', encoding='utf-8') as f:
            # default=str keeps an unexpected value (a date, a numpy scalar)
            # from aborting the write half-way through the file.
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    def _write_atomic(self, target, write_fn):
        """Write via a temp file in the same directory, then replace."""
        suffix = target.suffix
        tmp = target.with_name(f".{target.stem}.tmp{suffix}")
        try:
            write_fn(tmp)
            # np.save appends .npy when the path lacks it.
            if not tmp.exists() and tmp.with_suffix(tmp.suffix + '.npy').exists():
                tmp = tmp.with_suffix(tmp.suffix + '.npy')
            os.replace(str(tmp), str(target))
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def _quarantine_corrupt_file(self, file_path):
        """Move a corrupt store aside so the app can recover cleanly."""
        try:
            if not file_path.exists():
                return
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            quarantine_path = file_path.with_suffix(file_path.suffix + f'.bad_{stamp}')
            os.replace(str(file_path), str(quarantine_path))
            logger.info(f"Quarantined corrupt memory file: {quarantine_path}")
        except OSError as e:
            logger.warning(f"Could not quarantine corrupt memory file {file_path}: {e}")

    # ------------------------------------------------------------------
    # index / embedding helpers
    # ------------------------------------------------------------------

    @property
    def enabled(self):
        return self.embedder is not None

    def _encode(self, texts):
        """Embed a list of texts into a (n, dimension) float32 array."""
        if not texts or not self.enabled:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.asarray(self.embedder.encode(texts), dtype=np.float32)

    def _rebuild_index(self):
        """Rebuild the FAISS index from the embedding matrix."""
        if not faiss:
            self.index = None
            return
        self.index = faiss.IndexFlatL2(self.dimension)
        if self.embeddings is not None and len(self.embeddings):
            self.index.add(self.embeddings)

    def _rebuild_entity_map(self):
        self.entity_map = {}
        for position, record in enumerate(self.metadata):
            entity = record.get('entity', 'unknown')
            self.entity_map.setdefault(entity, []).append(position)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def search(self, query, top_k=3, user_key=None):
        """Search memory by semantic similarity, optionally scoped to one user."""
        if not self.enabled or not self.metadata or self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = self._encode([query or ''])
        query_tokens = set(re.findall(r"[a-z0-9]+", (query or '').lower()))

        # Over-fetch so that per-user filtering still has candidates to return.
        fetch = min(len(self.metadata), max(top_k * 5, top_k))
        distances, indices = self.index.search(query_embedding, fetch)

        results = []
        for rank, idx in enumerate(indices[0]):
            idx = int(idx)
            if idx < 0 or idx >= len(self.metadata):
                continue
            record = self.metadata[idx]
            if user_key is not None and record.get('user_key') != user_key:
                continue

            record_tokens = set(
                re.findall(r"[a-z0-9]+", f"{record.get('text', '')} {record.get('entity', '')}".lower())
            )
            lexical_boost = len(query_tokens & record_tokens) * 0.15
            entity = (record.get('entity') or '').lower()
            entity_match = 0.35 if entity and entity in (query or '').lower() else 0.0
            base_similarity = 1.0 / (1.0 + float(distances[0][rank]))

            results.append({
                'text': record['text'],
                'entity': record.get('entity'),
                'entity_type': record.get('entity_type'),
                'confidence': record.get('confidence', 1.0),
                'timestamp': record.get('timestamp'),
                'score': base_similarity + lexical_boost + entity_match,
            })

        results.sort(key=lambda item: item.get('score', 0), reverse=True)
        return results[:top_k]

    def store(self, text, entity, entity_type, confidence=0.9, user_key=None):
        """Store a memory, updating an existing record for the same entity.

        Returns True when the store was persisted.
        """
        text = (text or '').strip()
        if not text:
            return False
        if not self.enabled:
            logger.warning("Semantic memory is disabled; not storing %r", text)
            return False

        conflicts = self._detect_conflicts(entity, entity_type, user_key)
        if conflicts:
            for position in conflicts:
                self._resolve_conflict(position, text, confidence)
        else:
            self._add_memory(text, entity, entity_type, confidence, user_key)

        return self._save_memories()

    def _detect_conflicts(self, entity, entity_type, user_key=None):
        """Positions of existing memories describing the same entity."""
        conflicts = []
        for position in self.entity_map.get(entity, []):
            if position >= len(self.metadata):
                continue
            record = self.metadata[position]
            if record.get('entity_type') != entity_type:
                continue
            if user_key is not None and record.get('user_key') != user_key:
                continue
            conflicts.append(position)
        return conflicts

    def _resolve_conflict(self, position, new_text, new_confidence):
        """Update an existing memory in place, keeping its vector in step."""
        record = self.metadata[position]
        if new_confidence < record.get('confidence', 0):
            return
        if record.get('text') == new_text:
            return

        record['previous_value'] = record['text']
        record['text'] = new_text
        record['confidence'] = new_confidence
        record['timestamp'] = self._now()

        # Re-embed: without this the vector would still describe the old text
        # and the record would never be found by what it now says.
        self.embeddings[position] = self._encode([new_text])[0]
        self._rebuild_index()
        logger.info(f"Updated memory: {record['entity']} -> {new_text}")

    def _add_memory(self, text, entity, entity_type, confidence, user_key=None):
        """Append a new memory and its vector."""
        embedding = self._encode([text])

        record = {
            'id': len(self.metadata),
            'text': text,
            'entity': entity,
            'entity_type': entity_type,
            'confidence': confidence,
            'user_key': user_key,
            # An ISO string, not a datetime: the store is JSON and a datetime
            # cannot be serialised.
            'timestamp': self._now(),
        }

        self.metadata.append(record)
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = embedding
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])

        self.entity_map.setdefault(entity, []).append(len(self.metadata) - 1)
        self._rebuild_index()
        logger.info(f"Stored memory: {entity} -> {text}")

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()
