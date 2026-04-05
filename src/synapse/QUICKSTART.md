# Quick Start Guide

## 1. Setup (5 minutes)

```bash
cd src/synapse
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configure API Keys

```bash
# Copy template to .env
cp .env.example .env

# Edit .env with your keys:
# OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=...
```

## 3. Initialize Database

```bash
python manage.py migrate
```

## 4. Start Server

```bash
python manage.py runserver
```

- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:8000/voice/
- **WebSocket**: ws://localhost:8000/ws/voice/

## What's Running

- Django with Channels for WebSocket support
- 5 async workers:
  - **STT**: Whisper for speech-to-text
  - **Qwen**: Intent routing (fast decision making)
  - **Memory**: FAISS semantic retrieval
  - **GPT**: Reasoning for complex queries
  - **TTS**: OpenAI audio synthesis

## Testing

1. Open browser to http://localhost:8000/voice/
2. Click "Start Recording"
3. Say something like "Hello" or "What's the weather?"
4. Watch:
   - Transcript appear (partial → final)
   - Intent decision show up
   - Response stream in real-time
   - Audio play back

## Latency Targets

- First response visible: ~500ms
- Full response ready: ~2000ms
- Audio playing: ~3000ms

## Debugging

Check worker status:
```bash
# In another terminal:
curl http://localhost:8000/voice/status/
```

View logs:
```bash
# Watch server logs for [INFO] messages from each worker
```

## Next Steps

1. **Fine-tune Qwen** for your specific use cases
   - Collect intent examples
   - Use OUMI framework
   - Export and update QWEN_MODEL in settings.py

2. **Test Edge Cases**
   - Ambiguous queries
   - Memory conflicts
   - Fast back-to-back requests

3. **Deploy**
   - Move to production settings (DEBUG=False)
   - Use Daphne with proper concurrency
   - Enable CSRF protection
   - Set up proper logging

## Common Issues

### No audio output
- [ ] Check OPENAI_API_KEY is set
- [ ] Verify browser audio permissions
- [ ] Check network tab for WebSocket messages

### High latency
- [ ] Reduce Whisper buffer (currently 2 sec)
- [ ] Use GPU if available (set device="cuda" in workers)
- [ ] Check if GPT is overloaded

### Memory not storing
- [ ] Check FAISS directory exists
- [ ] Verify confidence scores from Qwen
- [ ] Check memory_worker logs
