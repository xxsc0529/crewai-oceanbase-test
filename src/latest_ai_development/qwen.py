"""
Qwen LLM for CrewAI agents (DashScope compatible API).
Reads API key and base URL from env: QWEN3_API_KEY or DASHSCOPE_API_KEY, BASE_URL, QWEN_MODEL.
"""
import os
from pathlib import Path

from crewai import LLM
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

QWEN_API_KEY = os.getenv("QWEN3_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
QWEN_BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
if "/v1" not in QWEN_BASE_URL:
    QWEN_BASE_URL = f"{QWEN_BASE_URL}/v1"
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-max")

qwen_llm = LLM(
    model=QWEN_MODEL,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    temperature=0.7,
    max_tokens=4096,
)
