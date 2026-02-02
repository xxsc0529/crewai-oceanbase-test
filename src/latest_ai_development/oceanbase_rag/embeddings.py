"""
DashScope text-embedding-v3 (1024 dim) via OpenAI-compatible API.
Used for PDF chunk vectors and OceanBaseVectorSearchTool query embedding.
"""
import os
from openai import OpenAI

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIM = 1024


def _get_client() -> OpenAI:
    api_key = os.getenv("QWEN3_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
    base_url = os.getenv("BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    if "/v1" not in base_url:
        base_url = f"{base_url}/v1"
    return OpenAI(api_key=api_key, base_url=base_url)


def get_embedding(text: str) -> list[float]:
    """Return 1024-dim embedding for text (DashScope text-embedding-v3)."""
    resp = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding
