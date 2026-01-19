"""Configuration module for reading environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv


# Load .env file
load_dotenv()


class Config:
    """Application configuration from environment variables."""
    
    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    
    # Paths
    DB_PATH: str = os.getenv("DB_PATH", "data/memory.db")
    COMPANY_TXT_PATH: str = os.getenv("COMPANY_TXT_PATH", "data/company.txt")
    
    # RAG Parameters
    TOP_K: int = int(os.getenv("TOP_K", "3"))
    RAG_THRESHOLD: float = float(os.getenv("RAG_THRESHOLD", "0.78"))
    
    # Logging
    DEBUG: bool = bool(int(os.getenv("DEBUG", "0")))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Short memory
    SHORT_MEMORY_SIZE: int = 10
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration parameters."""
        if not cls.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is not set in .env file")
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in .env file")
        
        # Check if company.txt exists
        if not Path(cls.COMPANY_TXT_PATH).exists():
            raise FileNotFoundError(f"Company data file not found: {cls.COMPANY_TXT_PATH}")
    
    @classmethod
    def get_db_dir(cls) -> Path:
        """Get database directory path."""
        return Path(cls.DB_PATH).parent
