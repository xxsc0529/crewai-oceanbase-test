# OceanBase RAG: PDF load, vector search, CrewAI RAG.
# Load project root .env and expose OceanBase config (no separate config file).
import os
from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_project_root / ".env")


def _e(k: str, d: str = "") -> str:
    return (os.getenv(k) or d).strip()


OCEANBASE_URI = f"{_e('OCEANBASE_HOST') or '127.0.0.1'}:{_e('OCEANBASE_PORT') or '2881'}"
OCEANBASE_USER = _e("OCEANBASE_USER") or "root"
OCEANBASE_PASSWORD = _e("OCEANBASE_PASSWORD") or ""
OCEANBASE_DB = _e("OCEANBASE_DB") or "crewai"
RAG_TABLE_NAME = "pdf_documents"
