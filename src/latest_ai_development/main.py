#!/usr/bin/env python
"""
Entry: PDF → OceanBase → CrewAI RAG. Commands: run, run_rag_cli, run_full_scenario_cli.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_QUERY = "文档中与收入、风险相关的主要内容有哪些？"


def run() -> None:
    """Run RAG with default query (uv tool run crewai run)."""
    from latest_ai_development.oceanbase_rag.crew import get_crew_response

    print(get_crew_response(DEFAULT_QUERY))


def run_rag(query: str) -> str:
    """RAG: semantic search + answer."""
    from latest_ai_development.oceanbase_rag.crew import get_crew_response

    return get_crew_response(query)


def run_rag_cli() -> None:
    """CLI: uv run rag_oceanbase [query]"""
    query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else DEFAULT_QUERY
    if not query:
        query = DEFAULT_QUERY
    print(run_rag(query))


def run_full_scenario(skip_load: bool = False, query: str = DEFAULT_QUERY) -> None:
    """Full flow: (optional) load PDF → vector search test → Crew RAG."""
    from latest_ai_development.oceanbase_rag.data_loader import OceanBaseData
    from latest_ai_development.oceanbase_rag.crew import get_crew_response

    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "knowledge" / "nke-10k-2023.pdf"

    if not skip_load:
        print("=" * 60)
        print("Step 1: PDF → DashScope embed → OceanBase")
        print("=" * 60)
        if not (os.getenv("QWEN3_API_KEY") or os.getenv("DASHSCOPE_API_KEY")):
            print("Error: set QWEN3_API_KEY or DASHSCOPE_API_KEY")
            sys.exit(1)
        if not pdf_path.is_file():
            print(f"Error: PDF not found: {pdf_path}")
            sys.exit(1)
        data = OceanBaseData()
        data.load_pdf(str(pdf_path))
        print()

    print("=" * 60)
    print("Step 2: Vector search test")
    print("=" * 60)
    data = OceanBaseData()
    print("Status:", data.check_status())
    results = data.search_documents(query, limit=3)
    print(f"Found {len(results)} hits:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. distance={r.get('distance')}, text={(r.get('text') or '')[:80]}...")
    print()

    print("=" * 60)
    print("Step 3: CrewAI RAG")
    print("=" * 60)
    result = get_crew_response(query)
    print("\n--- Final answer ---\n", result)


def run_full_scenario_cli() -> None:
    """CLI: uv run oceanbase_rag_scenario [--skip-load] [--query ...]"""
    parser = argparse.ArgumentParser(description="OceanBase RAG full scenario")
    parser.add_argument("--skip-load", action="store_true", help="Skip PDF load")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="RAG question")
    args = parser.parse_args()
    run_full_scenario(skip_load=args.skip_load, query=args.query)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("-"):
        run_full_scenario_cli()
    elif len(sys.argv) > 1:
        run_rag_cli()
    else:
        run()
