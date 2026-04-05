import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Build paths
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "faiss_memory"
FAISS_DIR.mkdir(exist_ok=True)

# Security
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = ['*']

# CSRF for development WebSocket (disable only for local dev)
CSRF_TRUSTED_ORIGINS = ['http://localhost:8000', 'http://127.0.0.1:8000']

# Apps
INSTALLED_APPS = [
    'daphne',
    'django.contrib.contenttypes',
    'django.contrib.auth',
    'django.contrib.sessions',
    'django.contrib.messages',
    'channels',
    'voice',
]

# Middleware
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
]

# ASGI Config
ASGI_APPLICATION = 'synapse.asgi.application'
WSGI_APPLICATION = 'synapse.wsgi.application'
ROOT_URLCONF = 'synapse.urls'

# Channels
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels.layers.InMemoryChannelLayer'
    }
}

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Templates
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            BASE_DIR / 'templates',
            BASE_DIR.parent.parent / 'FinalEclipse' / 'project' / 'synapse' / 'templates',
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{levelname}] {asctime} {name} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# API Keys from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY', '')

# Model settings
WHISPER_MODEL = "base"
QWEN_MODEL = os.getenv('OLLAMA_QWEN_MODEL', 'qwen2.5:3b-instruct')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434')
MISTRAL_MODEL = os.getenv('MISTRAL_MODEL', 'mistral-small-latest')
MISTRAL_BASE_URL = os.getenv('MISTRAL_BASE_URL', 'https://api.mistral.ai/v1')
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
USE_CUDA = os.getenv('USE_CUDA', 'true').lower() == 'true'

# Performance targets
LATENCY_FIRST_RESPONSE_MS = 500
LATENCY_FULL_RESPONSE_MS = 3000
