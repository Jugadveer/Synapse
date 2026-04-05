# Implementation Checklist

## Phase 1: Core Architecture ✅

- [x] Django + Channels setup
- [x] WebSocket consumer
- [x] Database models (Session, Memory, Turn)
- [x] Async pipeline orchestration
- [x] 5 worker modules (STT, Qwen, Memory, GPT, TTS)
- [x] FAISS memory system with conflict resolution
- [x] Configuration management
- [x] Environment-based secrets (.env)
- [x] Frontend HTML/JS with WebSocket client

## Phase 2: Testing & Validation

- [ ] **Unit Tests**
  - [ ] STT worker transcription accuracy
  - [ ] Qwen intent classification on 10+ examples
  - [ ] Memory storage and retrieval
  - [ ] Conflict detection (same entity, different values)
  - [ ] GPT response generation

- [ ] **Integration Tests**
  - [ ] Full pipeline: audio → text → intent → response → audio
  - [ ] WebSocket message ordering
  - [ ] Database persistence
  - [ ] FAISS indexing and recall

- [ ] **Latency Profiling**
  - [ ] Record latencies for each stage
  - [ ] Identify bottlenecks
  - [ ] Optimize slowest worker

## Phase 3: Dementia-Safe Features

- [ ] **Soft Updates**
  - [ ] Log conflicting memory updates
  - [ ] Present alternatives ("I have you as X, did you mean Y?")
  - [ ] Never force-overwrite without confirmation

- [ ] **Interaction Patterns**
  - [ ] Warm greeting on first message
  - [ ] Repeated info tolerance (no correction)
  - [ ] Validation before memory updates

- [ ] **Accessibility**
  - [ ] Large text option (CSS)
  - [ ] High contrast mode
  - [ ] Slow speech playback option

## Phase 4: Model Fine-Tuning

- [ ] **Qwen Fine-Tuning Dataset**
  - [ ] 100+ intent examples (dementia-specific)
  - [ ] Edge cases: ambiguity, repetition, emotional
  - [ ] Expected JSON outputs for each

- [ ] **Fine-Tuning via OUMI**
  - [ ] Setup OUMI environment
  - [ ] Create training config
  - [ ] Run training (watch GPU memory)
  - [ ] Export to HuggingFace model format

- [ ] **Validation**
  - [ ] Test on holdout set
  - [ ] Compare to base model
  - [ ] Deploy to production

## Phase 5: Deployment

- [ ] **Production Hardening**
  - [ ] Set DEBUG=False
  - [ ] Use Daphne with gunicorn
  - [ ] Enable CSRF protection
  - [ ] Add rate limiting

- [ ] **Monitoring**
  - [ ] Latency tracking (all stages)
  - [ ] Error logs (per worker)
  - [ ] Memory usage (FAISS index growth)
  - [ ] Audio quality metrics

- [ ] **Scaling**
  - [ ] Multiple Daphne workers
  - [ ] Redis for channel layers
  - [ ] Load balancing (nginx)
  - [ ] Database optimization (add indexes)

## Phase 6: Documentation & Handoff

- [ ] README with setup instructions
- [ ] Architecture diagram
- [ ] API documentation (WebSocket protocol)
- [ ] Configuration reference
- [ ] Troubleshooting guide
- [ ] Model fine-tuning guide

---

## Immediate Next Tasks (Priority Order)

### **NOW: Verify System Runs**
1. ✅ Create pipeline architecture (DONE)
2. ✅ Create worker modules (DONE)
3. ✅ Create frontend UI (DONE)
4. [ ] **RUN**: `python manage.py migrate`
5. [ ] **RUN**: `python manage.py runserver`
6. [ ] **TEST**: Open http://localhost:8000/voice/ and speak

### **THEN: Fix Issues**
- [ ] Debug worker import errors
- [ ] Verify WebSocket connection
- [ ] Check STT model downloads automatically
- [ ] Verify API keys work

### **THEN: Profile Performance**
- [ ] Measure each worker's latency
- [ ] Identify bottleneck
- [ ] Optimize (quantization, batch size, device)

### **THEN: Add Dementia Features**
- [ ] Soft memory updates
- [ ] Conflict resolution UI
- [ ] Repeated query handling
- [ ] Warmth in responses

### **THEN: Fine-Tune Qwen**
- [ ] Collect 100+ real examples
- [ ] Use OUMI to train
- [ ] Deploy fine-tuned model
- [ ] Validate on test set

---

## Files Created

### Core
- [x] `synapse/settings.py` - Django config
- [x] `synapse/asgi.py` - Channels entry point
- [x] `synapse/urls.py` - URL routing
- [x] `manage.py` - Management script

### Voice App
- [x] `voice/models.py` - Database models
- [x] `voice/consumers.py` - WebSocket handler
- [x] `voice/views.py` - HTTP endpoints
- [x] `voice/routing.py` - WebSocket routing
- [x] `voice/urls.py` - HTTP routing

### Pipeline
- [x] `pipeline/pipeline.py` - Main orchestrator
- [x] `pipeline/stt_worker.py` - Whisper
- [x] `pipeline/qwen_router.py` - Intent routing
- [x] `pipeline/gpt_worker.py` - GPT reasoning
- [x] `pipeline/memory_worker.py` - Memory management
- [x] `pipeline/tts_worker.py` - Audio synthesis

### Models & Utils
- [x] `models_wrapper/faiss_memory.py` - Vector DB
- [x] `utils/config.py` - Configuration
- [x] `utils/prompts.py` - Dementia-aware prompts

### Frontend & Config
- [x] `templates/voice/index.html` - WebSocket UI
- [x] `requirements.txt` - Dependencies
- [x] `.env.example` - Template
- [x] `README_DJANGO.md` - Architecture docs
- [x] `QUICKSTART.md` - Setup guide

---

## Known Limitations (TODO)

- [ ] Qwen runs on CPU (slow) - need GPU or quantization
- [ ] GPT streaming not yet integrated with response display
- [ ] Memory conflict UI not fully implemented
- [ ] No persistence between sessions (can add to models)
- [ ] Audio quality: Whisper `base` might need `small` for accuracy
- [ ] TTS: Not streaming to client yet (full response sent at once)

---

## Success Criteria

- [x] Code structure complete
- [ ] System runs without errors
- [ ] First response within 500ms
- [ ] Full response within 3 seconds
- [ ] Audio output quality acceptable
- [ ] Memory retrieval working
- [ ] Conflict detection working
