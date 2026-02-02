"""
OceanBase data loader: extract PDF text, embed with DashScope, store in OceanBase.
Provides load_pdf(), search_documents(), and check_status() for the RAG pipeline.
"""
import re
from pathlib import Path

import pdfplumber
from pyobvector import (
    ObVecClient,
    VECTOR,
    IndexParams,
    VecIndexType,
    cosine_distance,
)
from sqlalchemy import Column, Integer, String
from sqlalchemy import text as sql_text

from latest_ai_development.oceanbase_rag import (
    OCEANBASE_DB,
    OCEANBASE_PASSWORD,
    OCEANBASE_URI,
    OCEANBASE_USER,
    RAG_TABLE_NAME,
)
from latest_ai_development.oceanbase_rag.embeddings import EMBEDDING_DIM, get_embedding

# Chunk size in characters for splitting PDF text (roughly 500 tokens)
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def _extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            # Try to break at sentence or space
            break_at = text.rfind(". ", start, end + 1)
            if break_at == -1:
                break_at = text.rfind(" ", start, end + 1)
            if break_at != -1:
                end = break_at + 1
        chunks.append(text[start:end].strip())
        start = end - CHUNK_OVERLAP
        if start < 0:
            start = end
    return [c for c in chunks if c]


class OceanBaseData:
    """
    Load PDF into OceanBase (vector table) and run vector search.
    Table: id (PK), text, embedding VECTOR(1024) with HNSW index.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        db_name: str | None = None,
        table_name: str | None = None,
    ):
        self.uri = uri or OCEANBASE_URI
        self.user = user or OCEANBASE_USER
        self.password = password or OCEANBASE_PASSWORD
        self.db_name = db_name or OCEANBASE_DB
        self.table_name = table_name or RAG_TABLE_NAME
        self._client: ObVecClient | None = None

    def _get_client(self) -> ObVecClient:
        """Lazy ObVecClient connection."""
        if self._client is None:
            self._client = ObVecClient(
                uri=self.uri,
                user=self.user,
                password=self.password,
                db_name=self.db_name,
            )
        return self._client

    def load_pdf(self, pdf_path: str) -> None:
        """
        Extract text from PDF, chunk, embed with DashScope, and insert into OceanBase.
        Recreates the table if it exists (drop + create).
        """
        client = self._get_client()
        # Drop existing table so we can recreate with same schema
        client.drop_table_if_exist(self.table_name)
        client.refresh_metadata()

        # Table: id, text, embedding (VECTOR 1024)
        columns = [
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("text", String(16000)),
            Column("embedding", VECTOR(EMBEDDING_DIM)),
        ]
        vidxs = IndexParams()
        vidxs.add_index(
            field_name="embedding",
            index_type=VecIndexType.HNSW,
            index_name="pdf_embedding_idx",
        )
        client.create_table_with_index_params(
            self.table_name,
            columns=columns,
            vidxs=vidxs,
        )
        client.refresh_metadata(tables=[self.table_name])

        raw_text = _extract_text_from_pdf(pdf_path)
        chunks = _chunk_text(raw_text)
        if not chunks:
            return

        # Insert in batches (embedding API may have rate limits)
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            rows = []
            for j, chunk in enumerate(batch):
                row_id = i + j + 1
                vec = get_embedding(chunk)
                rows.append({"id": row_id, "text": chunk[:16000], "embedding": vec})
            client.insert(self.table_name, rows)

    def check_status(self) -> str:
        """Return a short status string (e.g. row count) for the vector table."""
        client = self._get_client()
        if not client.check_table_exists(self.table_name):
            return "table does not exist"
        with client.engine.connect() as conn:
            r = conn.execute(sql_text(f"SELECT COUNT(*) FROM `{self.table_name}`"))
            count = r.scalar()
        return f"rows={count}"

    def search_documents(self, query: str, limit: int = 5) -> list[dict]:
        """
        Vector search: embed query, then ANN search in OceanBase.
        Returns list of dicts with keys: distance, text, etc.
        """
        client = self._get_client()
        if not client.check_table_exists(self.table_name):
            return []
        query_vec = get_embedding(query)
        # ann_search returns a result proxy; fetch rows
        result = client.ann_search(
            table_name=self.table_name,
            vec_data=query_vec,
            vec_column_name="embedding",
            distance_func=cosine_distance,
            with_dist=True,
            topk=limit,
            output_column_names=["id", "text"],
        )
        rows = result.fetchall()
        # Row format: (id, text, distance) when output_column_names + with_dist
        out = []
        for row in rows:
            r = list(row)
            dist = r[-1] if len(r) > 2 else None
            text_val = r[1] if len(r) > 1 else ""
            out.append({"distance": dist, "text": text_val})
        return out


def _project_root() -> Path:
    """Project root (parent of src)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def load_pdf_oceanbase(pdf_path: str | Path | None = None) -> int:
    """
    Load PDF into OceanBase (CLI: uv run load_pdf_oceanbase).
    Returns 0 on success, 1 on error.
    """
    from dotenv import load_dotenv

    load_dotenv(_project_root() / ".env")
    path = str(pdf_path) if pdf_path else str(_project_root() / "knowledge" / "nke-10k-2023.pdf")
    if not Path(path).is_file():
        print(f"Error: PDF not found: {path}")
        return 1
    data = OceanBaseData()
    data.load_pdf(path)
    print("Loaded PDF into OceanBase. Status:", data.check_status())
    return 0


def main() -> None:
    """CLI: ensure table exists (load default PDF if empty), then run one search and print."""
    from dotenv import load_dotenv

    load_dotenv(_project_root() / ".env")
    data = OceanBaseData()
    default_pdf = _project_root() / "knowledge" / "nke-10k-2023.pdf"
    status = data.check_status()
    if "rows=0" in status or "does not exist" in status:
        if default_pdf.is_file():
            data.load_pdf(str(default_pdf))
        else:
            print("No PDF at", default_pdf)
            return
    print("Status:", data.check_status())
    query = "What are the main risks or revenues mentioned?"
    for r in data.search_documents(query, limit=3):
        print(r.get("distance"), (r.get("text") or "")[:120])


if __name__ == "__main__":
    main()
