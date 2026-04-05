import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """Centralized configuration management."""
    
    # Model settings
    WHISPER_MODEL = "base"
    QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
    EMBEDDER_MODEL = "all-MiniLM-L6-v2"
    
    # API keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    FAISS_DIR = PROJECT_ROOT / "faiss_memory"
    TEMPLATES_DIR = PROJECT_ROOT / "templates"
    
    # Performance
    STT_BUFFER_SIZE = 32000
    STT_SILENCE_TIMEOUT = 1.0
    LLM_MAX_TOKENS = 100
    TTS_CHUNK_SIZE = 8192
    
    # Latency targets (ms)
    TARGET_FIRST_RESPONSE = 500
    TARGET_FULL_RESPONSE = 3000
    
    @classmethod
    def validate(cls):
        """Validate critical configuration."""
        if not cls.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set; TTS disabled")
        if not cls.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set")
