import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.8-flash")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    database_path: Path = Path(os.getenv("DATABASE_PATH", "./data/inventory.db"))
    approval_threshold: float = float(os.getenv("APPROVAL_THRESHOLD", "0"))
    policy_store_path: Path = Path(os.getenv("POLICY_STORE_PATH", "./data/policy_index"))
    policy_folder: Path = Path(os.getenv("POLICY_FOLDER", "./data/policies"))
