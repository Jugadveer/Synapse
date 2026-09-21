"""A stand-in Ollama server for end-to-end tests.

Serves /api/generate over real HTTP on a real port, so the router's httpx
client, JSON parsing and error handling are all genuinely exercised. Only the
model weights are substituted: responses are scripted from the prompt text.

Install Ollama and pull the router model to run the same tests against the
real thing - nothing else in the pipeline changes.
"""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Matches the last quoted line of either prompt template.
_USER_LINE = re.compile(r'User(?: message)?: "(.*)"\s*$', re.MULTILINE)

OBJECTS = ('keys', 'wallet', 'glasses', 'phone', 'tablets', 'medicine', 'book')
PLACES = ('table', 'drawer', 'shelf', 'counter', 'kitchen', 'bedroom', 'sofa', 'bag')


def _user_text(prompt):
    matches = _USER_LINE.findall(prompt)
    return matches[-1] if matches else ''


def _mentions(text, terms):
    return next((t for t in terms if re.search(rf'\b{t}\b', text.lower())), '')


def memory_reply(prompt):
    """Scripted memory-analyst answer, following the schema in prompts.py."""
    text = _user_text(prompt)
    lowered = text.lower()
    obj = _mentions(text, OBJECTS)
    place = _mentions(text, PLACES)

    asking = bool(re.search(r'\b(where|what|which|when|do you remember)\b', lowered))
    if asking:
        return {
            'intent': 'memory_retrieve', 'is_fast': True, 'needs_memory': True,
            'needs_reasoning': False, 'needs_memory_storage': False,
            'needs_memory_retrieval': True, 'needs_clarification': False,
            'information_completeness': {'is_complete': True, 'missing_fields': [], 'should_ask': False},
            'memory_entity': obj, 'memory_entity_type': 'location', 'memory_value': '',
            'memory_query': obj or text, 'clarification_question': '', 'confidence': 0.94,
        }

    stating = bool(re.search(r'\b(i (left|put|kept|placed)|my|remember)\b', lowered))
    if stating and obj:
        complete = bool(place)
        return {
            'intent': 'memory_store', 'is_fast': True, 'needs_memory': True,
            'needs_reasoning': False, 'needs_memory_storage': complete,
            'needs_memory_retrieval': False, 'needs_clarification': not complete,
            'information_completeness': {
                'is_complete': complete,
                'missing_fields': [] if complete else ['location'],
                'should_ask': not complete,
            },
            'memory_entity': obj, 'memory_entity_type': 'location',
            'memory_value': text, 'memory_query': '',
            'clarification_question': '' if complete else 'Where did you put them?',
            'confidence': 0.93 if complete else 0.6,
        }

    return {
        'intent': 'other', 'is_fast': True, 'needs_memory': False, 'needs_reasoning': False,
        'needs_memory_storage': False, 'needs_memory_retrieval': False,
        'needs_clarification': False,
        'information_completeness': {'is_complete': True, 'missing_fields': [], 'should_ask': False},
        'memory_entity': '', 'memory_entity_type': 'fact', 'memory_value': '',
        'memory_query': '', 'clarification_question': '', 'confidence': 0.9,
    }


def classify_reply(prompt):
    """Scripted classification answer."""
    text = _user_text(prompt)
    lowered = text.lower()

    if re.search(r'\b(hello|hi|good morning|thank you|how are you)\b', lowered):
        return {
            'intent': 'casual', 'is_fast': True, 'needs_memory': False,
            'needs_reasoning': False, 'fast_response': 'Hello. It is good to hear from you.',
            'memory_query': '', 'memory_content': '', 'confidence': 0.95,
        }

    if re.search(r'\b(tell me about|explain|why|how should|what should)\b', lowered):
        return {
            'intent': 'question', 'is_fast': False, 'needs_memory': False,
            'needs_reasoning': True, 'fast_response': '', 'memory_query': '',
            'memory_content': '', 'confidence': 0.88,
        }

    return {
        'intent': 'unclear', 'is_fast': True, 'needs_memory': False,
        'needs_reasoning': False, 'fast_response': 'Could you tell me a bit more?',
        'memory_query': '', 'memory_content': '', 'confidence': 0.3,
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the test output quiet
        pass

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length) or b'{}')
        prompt = body.get('prompt', '')

        self.server.prompts.append(prompt)

        if body.get('format') == 'json':
            reply = (memory_reply if 'memory-intent analyst' in prompt else classify_reply)(prompt)
            response = json.dumps(reply)
        else:
            # Free-text calls: acknowledgements and retrieval phrasing.
            response = self.server.text_reply

        payload = json.dumps({'response': response, 'done': True}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class FakeOllama:
    """Context manager yielding a running server and its base URL."""

    def __init__(self, text_reply='Of course, I have made a note of that.'):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
        self.server.prompts = []
        self.server.text_reply = text_reply
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        host, port = self.server.server_address[:2]
        return f'http://{host}:{port}'

    @property
    def prompts(self):
        return self.server.prompts

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
