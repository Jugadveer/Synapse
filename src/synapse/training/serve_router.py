"""Serve the fine-tuned router over Ollama's /api/generate contract.

Ollama stopped accepting LoRA adapters ("LoRA adapters are no longer
supported" as of 0.34), and converting to GGUF needs the llama.cpp toolchain.
The router only ever calls one endpoint, so serving the merged model directly
is smaller than either: no extra dependencies, no download, and the
application needs no change beyond pointing OLLAMA_URL at this process.

    python merge_router.py
    python serve_router.py --port 11500
    # then in .env:  OLLAMA_URL=http://127.0.0.1:11500

Stay on Ollama instead by leaving OLLAMA_URL alone; the base model still
works, just less accurately (4/8 against 8/8 on the spoken cases).

Unload Ollama's own model first. Both want the same card, and with under a
gigabyte spare the allocator thrashes: the same turn that takes two seconds
with room takes twenty to fifty without it.

    ollama stop qwen2.5:1.5b-instruct
"""

import argparse
import json
import logging
import os
import re
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('router')

HERE = Path(__file__).resolve().parent
MERGED = HERE / 'artifacts' / 'router-merged'
#: The model emits end-of-sequence after 50 to 110 tokens on its own, so this
#: is a backstop rather than the usual stopping point.
MAX_NEW_TOKENS = 256

#: Below this much free VRAM the CUDA allocator starts thrashing and each turn
#: takes 20 to 50 seconds instead of two to five. Measured with Ollama holding
#: its own model on the same 6 GB card.
HEADROOM_BYTES = 1_200_000_000

#: Generation is serialised: one model, and concurrent generate() calls on the
#: same weights contend for the device rather than going faster.
_lock = threading.Lock()
_model = None
_tokenizer = None
#: Two system prompts. Sending the JSON one for a free-text request told the
#: model to answer in JSON and then the caller parsed it as prose.
_JSON_SYSTEM = 'You are a decision layer for a dementia-safe voice assistant. Output JSON only.'
_TEXT_SYSTEM = (
    'You are a warm, concise assistant for a person living with dementia. '
    'Answer in one short sentence.'
)


def load(path, device):
    global _model, _tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f'loading {path}')
    _tokenizer = AutoTokenizer.from_pretrained(str(path))
    _model = AutoModelForCausalLM.from_pretrained(
        str(path),
        dtype=torch.float16 if device == 'cuda' else torch.float32,
        # Measured faster than loading then .to('cuda'): 21.7 against 17.0
        # tokens a second, for the same memory.
        device_map='cuda' if device == 'cuda' else None,
    ).eval()
    if device != 'cuda':
        _model = _model.to('cpu')

    if device == 'cuda':
        free, total = torch.cuda.mem_get_info()
        logger.info(f'ready on {device}, {free / 1e9:.1f} GB of {total / 1e9:.1f} GB free')
        if free < HEADROOM_BYTES:
            logger.warning(
                'Less than %.1f GB of VRAM free. Generation slows by roughly ten '
                'times when the allocator runs out of room - the symptom is 20 to '
                '50 seconds a turn instead of two to five. Free some up with: '
                'ollama stop <model>',
                HEADROOM_BYTES / 1e9,
            )
    else:
        logger.info(f'ready on {device}')


@torch.inference_mode()
def generate(prompt, temperature, want_json):
    messages = [
        {'role': 'system', 'content': _JSON_SYSTEM if want_json else _TEXT_SYSTEM},
        {'role': 'user', 'content': prompt},
    ]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(text, return_tensors='pt').to(_model.device)
    prompt_length = inputs['input_ids'].shape[1]

    kwargs = {
        'max_new_tokens': MAX_NEW_TOKENS,
        'pad_token_id': _tokenizer.pad_token_id or _tokenizer.eos_token_id,
    }
    # Greedy for a decision layer: the same turn should route the same way.
    if temperature > 0.1:
        kwargs.update(do_sample=True, temperature=temperature)
    else:
        kwargs.update(do_sample=False)

    output = _model.generate(**inputs, **kwargs)
    reply = _tokenizer.decode(
        output[0][prompt_length:], skip_special_tokens=True
    ).strip()

    if want_json:
        # The caller parses with a {...} search, but hand back clean JSON when
        # it can be isolated, so a stray preamble cannot confuse it.
        match = re.search(r'\{.*\}', reply, re.DOTALL)
        if match:
            try:
                return json.dumps(json.loads(match.group(0)))
            except json.JSONDecodeError:
                return match.group(0)
    return reply


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # Enough of Ollama's surface for a health check to pass.
        if self.path.startswith('/api/version'):
            return self._send({'version': 'synapse-router'})
        if self.path.startswith('/api/tags'):
            return self._send({'models': [{'name': 'synapse-router', 'model': 'synapse-router'}]})
        self._send({'error': 'not found'}, status=404)

    def do_POST(self):
        if not self.path.startswith('/api/generate'):
            return self._send({'error': 'not found'}, status=404)

        try:
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length) or b'{}')
        except (ValueError, json.JSONDecodeError):
            return self._send({'error': 'bad request'}, status=400)

        prompt = body.get('prompt', '')
        if not isinstance(prompt, str) or not prompt.strip():
            return self._send({'error': 'prompt required'}, status=400)

        options = body.get('options') or {}
        temperature = float(options.get('temperature', 0.1))
        want_json = body.get('format') == 'json'

        try:
            with _lock:
                reply = generate(prompt, temperature, want_json)
        except Exception as exc:
            logger.exception('generation failed')
            return self._send({'error': str(exc)}, status=500)

        self._send({'model': 'synapse-router', 'response': reply, 'done': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(MERGED))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=11500)
    parser.add_argument('--cpu', action='store_true', help='force CPU even with a GPU present')
    args = parser.parse_args()

    path = Path(args.model)
    if not path.exists():
        raise SystemExit(f'No model at {path}. Run merge_router.py first.')

    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    load(path, device)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info(f'listening on http://{args.host}:{args.port}')
    logger.info(f'point OLLAMA_URL at it: OLLAMA_URL=http://{args.host}:{args.port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info('stopping')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
