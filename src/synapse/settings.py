import os
from pathlib import Path
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config('DJANGO_SECRET_KEY', default='dev-secret-key-change-in-production')
DEBUG = config('DEBUG', default=True, cast=bool)
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'daphne',
    'django.contrib.contenttypes',
    'django.contrib.auth',
    'django.contrib.sessions',
    'django.contrib.messages',
    'channels',
    'voice',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
]

ROOT_URLCONF = 'synapse.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            BASE_DIR / 'synapse' / 'templates',
            BASE_DIR.parent / 'FinalEclipse' / 'project' / 'synapse' / 'templates',
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

WSGI_APPLICATION = 'synapse.wsgi.application'
ASGI_APPLICATION = 'synapse.asgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Channels configuration
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels.layers.InMemoryChannelLayer'
    }
}

# Application settings
CHUNK_SIZE = 16000  # 1 second at 16kHz
PARTIAL_TRANSCRIPTION_TIMEOUT = 1.0
TTL_INTENT_CACHE = 300
FAISS_INDEX_PATH = BASE_DIR / 'static' / 'faiss_index'
MEMORY_DB_PATH = BASE_DIR / 'static' / 'memory.db'

# API Keys
GEMINI_API_KEY = config('GEMINI_API_KEY', default='')
OPENAI_API_KEY = config('OPENAI_API_KEY', default='')
HUGGINGFACE_TOKEN = config('HF_TOKEN', default='')

# Model paths
QWEN_MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct'
WHISPER_MODEL = 'base'
