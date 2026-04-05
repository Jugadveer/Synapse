# Synapse Voice Assistant - Django Architecture

## Project Structure

```
synapse_agent/
├── src/synapse/
│   ├── manage.py                 # Django management script
│   ├── requirements.txt           # Python dependencies
│   ├── .env                       # Environment variables (local dev)
│   ├── .env.example              # Template for .env
│   ├── db.sqlite3                # SQLite database (auto-created)
│   │
│   ├── synapse/                  # Django project config
│   │   ├── __init__.py
│   │   ├── settings.py           # Django configuration
│   │   ├── asgi.py              # ASGI app entry (Channels)
│   │   ├── wsgi.py              # WSGI app entry
│   │   └── urls.py              # URL routing
│   │
│   ├── voice/                    # Voice assistant app
│   │   ├── migrations/           # Database migrations
│   │   ├── models.py            # Django models (Session, Memory, Turn)
│   │   ├── consumers.py         # WebSocket consumer
│   │   ├── views.py             # HTTP views (/status/)
│   │   ├── urls.py              # HTTP routing
│   │   ├── routing.py           # WebSocket routing
│   │   └── __init__.py
│   │
│   ├── pipeline/                 # Async pipeline orchestration
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Main AsyncPipeline orchestrator
│   │   ├── stt_worker.py        # Whisper STT worker
│   │   ├── qwen_router.py       # Qwen intent routing worker
│   │   ├── gpt_worker.py        # GPT-4o Mini reasoning worker
│   │   ├── memory_worker.py     # Memory retrieval & storage
│   │   └── tts_worker.py        # OpenAI TTS worker
│   │
│   ├── models_wrapper/           # Model interfaces
│   │   ├── __init__.py
│   │   └── faiss_memory.py      # FAISS vector memory with conflict resolution
│   │
│   ├── utils/                    # Utilities
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration manager
│   │   └── prompts.py           # Dementia-aware prompts
│   │
│   ├── templates/
│   │   └── voice/
│   │       └── index.html       # Frontend UI (WebSocket client)
│   │
│   └── faiss_memory/             # Persistent vector storage
│       ├── metadata.json        # Memory metadata
│       └── index.faiss          # FAISS index file
│
└── (legacy) backend/            # Original FastAPI implementation
```

## Architecture Overview

### **Multi-Worker Async Pipeline**

```
Audio Input (WebSocket)
    ↓
[STT Worker] - faster-whisper (base model)
    ↓ (text chunks)
[Qwen Router] - Qwen2.5-3B intent classification
    ↓ (decision: needs_gpt? memory_action?)
    ├─→ [Fast Response] → TTS
    │
    └─→ [GPT Worker] - GPT-4o Mini (streaming, dementia-aware)
        ├─→ [Memory Worker] - FAISS retrieval + storage
        └─→ [TTS Worker] - OpenAI text-to-speech
            ↓
        Audio Output (WebSocket)
```

### **Key Design Decisions**

1. **Qwen for Routing, GPT for Reasoning**
   - Qwen makes fast decisions (intent, memory needs) → no latency from LLM startup
   - GPT only called when complex reasoning needed
   - Saves ~300ms on simple queries

2. **Queue-Based Workers**
   - Decoupled workers via asyncio queues
   - No blocking callbacks
   - Easy to add/remove workers or adjust concurrency

3. **Django Models for Structured Memory**
   - `ConversationSession` - session tracking
   - `MemoryRecord` - entities with conflict resolution
   - `ConversationTurn` - logging for analysis

4. **FAISS for Semantic Retrieval**
   - Fast vector search in real-time
   - Soft conflict resolution for dementia safety
   - Persistent storage with hot reload

5. **Dementia-Safe Defaults**
   - Short responses (1-2 sentences)
   - Never contradict confirmed memories
   - Validation before memory updates
   - Warm, reassuring tone

## Setup Instructions

### 1. Install Dependencies

```bash
cd src/synapse
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` → `.env` and populate:

```env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
SECRET_KEY=your-secret-key
DEBUG=True
```

### 3. Initialize Database

```bash
python manage.py migrate
```

### 4. Run Development Server

```bash
python manage.py runserver 0.0.0.0:8000
```

The server runs on `http://localhost:8000` with:
- WebSocket: `ws://localhost:8000/ws/voice/`
- HTTP Status: `GET /voice/status/`
- Frontend: `http://localhost:8000/voice/`

## API Contract

### WebSocket Messages (to server)

```json
{
  "type": "interrupt"
}
```

Binary audio chunks (raw PCM bytes) are also accepted.

### WebSocket Messages (from server)

```json
{
  "type": "transcript",
  "text": "Hello",
  "is_final": false
}

{
  "type": "decision",
  "intent": "greeting",
  "needs_gpt": false,
  "confidence": 0.95
}

{
  "type": "response_chunk",
  "text": "Hi there!"
}

{
  "type": "memory_update",
  "action": "create",
  "entity": "Alice",
  "value": "prefers tea"
}

{
  "type": "latency",
  "latency_ms": 450
}
```

## Worker Details

### STT Worker (faster-whisper)
- Input: audio PCM bytes
- Output: partial transcripts at 2-second intervals, final on silence
- Latency: ~200-400ms

### Qwen Router (Qwen2.5-3B)
- Input: transcribed text
- Decision fields:
  - `intent_type`: greeting | command | question | memory_query | memory_store | casual
  - `is_complex`: boolean (needs GPT?)
  - `memory_action`: none | retrieve | store
  - `quick_response`: fallback for simple queries
  - `confidence`: 0.0-1.0

### GPT Worker (GPT-4o Mini)
- Input: complex queries requiring reasoning
- Output: streaming text chunks
- Constraints: max 100 tokens, dementia-safe language

### Memory Worker
- Input: `memory_query` or `memory_store` from Qwen
- FAISS search: semantic retrieval by query
- Conflict detection: same entity, different values
- Resolution: prefer higher confidence, more recent

### TTS Worker (OpenAI)
- Input: complete response text
- Output: MP3 audio stream
- Voice: "nova" (warm, natural)
- Parameters: model="tts-1", speed=1.0

## Latency Targets

| Stage | Target | Notes |
|-------|--------|-------|
| STT → Qwen | 500ms | First "thinking" response |
| Qwen → GPT | 1000ms | First token from GPT |
| GPT → TTS | 2000ms | Complete response ready |
| TTS → Audio | 3000ms | User hears response |

## Model Configuration

### Whisper
- Model: `base` (140M params, good accuracy/speed tradeoff)
- Device: CPU with int8 quantization
- VAD Filter: enabled (skip silence)

### Qwen
- Model: `Qwen/Qwen2.5-3B-Instruct` (fine-tuned via OUMI)
- Device: CPU (can be moved to CUDA if available)
- Temperature: 0.3 (low - deterministic routing)
- Max tokens: 200

### GPT
- Model: `gpt-4o-mini` (fast, cost-effective)
- Stream: enabled (partial response for UX)
- Max tokens: 100 (dementia-safe)
- Temperature: 0.7 (balanced)

### Embeddings
- Model: `all-MiniLM-L6-v2` (384-dim, efficient)
- Batch size: 32

## Frontend Features

- Real-time audio waveform visualization
- Partial + final transcription display
- Instant decision transparency (intent, confidence)
- Streaming response text
- Audio playback with status
- Memory updates log
- Turn latency metrics

## Development Notes

### Adding a New Worker

1. Create `pipeline/{worker_name}.py` with `{WorkerName}` class
2. Implement `async def run(self)` method
3. Connect input queue in `AsyncPipeline.__init__`
4. Update consumer to call `send_{worker_name}()` if needed

### Fine-Tuning Qwen

- Use OUMI framework: `https://github.com/gair-nlp/OUMI`
- Training data: intent examples with expected JSON output
- Export as HuggingFace model after training
- Update `QWEN_MODEL` path in settings.py

### Deployment Checklist

- [ ] Set all API keys in `.env`
- [ ] Run migrations: `python manage.py migrate`
- [ ] Test WebSocket: open Browser DevTools → Network → WS
- [ ] Monitor latency metrics from `/voice/status/`
- [ ] Enable CSRF protection for production
- [ ] Use Daphne with proper concurrency settings
- [ ] Set DEBUG=False

## Troubleshooting

**WebSocket connects but gets no transcript**
- Check Whisper model is loaded (GPU memory?)
- Verify audio stream is actually being sent
- Check STT Worker logs for errors

**High latency (>2000ms)**
- STT: use quantized Whisper, reduce buffer size
- Qwen: use INT8 quantization, smaller batch
- GPT: stream partial responses
- Debug with `/voice/status/` endpoint

**Memory conflicts not resolving**
- Check FAISS embedding similarity threshold
- Verify confidence scores are coherent
- Log memory_worker for conflict detection

**Frontend audio playback laggy**
- Reduce chunk size from server
- Use streaming with chunked playback
- Check browser audio buffer settings
