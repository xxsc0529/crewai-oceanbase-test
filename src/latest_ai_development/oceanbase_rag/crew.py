"""
CrewAI RAG crew: semantic search over OceanBase + answer synthesis.
Uses OceanBaseVectorSearchTool with DashScope embedding; agents/tasks from YAML.
"""
from pathlib import Path

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import OceanBaseVectorSearchTool
from crewai_tools.tools.oceanbase_vector_search_tool import OceanBaseConfig

from latest_ai_development.oceanbase_rag import (
    OCEANBASE_DB,
    OCEANBASE_PASSWORD,
    OCEANBASE_URI,
    OCEANBASE_USER,
    RAG_TABLE_NAME,
)
from latest_ai_development.oceanbase_rag.embeddings import get_embedding
from latest_ai_development.qwen import qwen_llm

# Config directory (agents.yaml, tasks.yaml)
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_yaml(name: str) -> dict:
    """Load agents or tasks YAML from config dir."""
    path = _CONFIG_DIR / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_oceanbase_tool() -> OceanBaseVectorSearchTool:
    """Build OceanBase vector search tool with DashScope embedding (same as data_loader)."""
    config = OceanBaseConfig(
        uri=OCEANBASE_URI,
        user=OCEANBASE_USER,
        password=OCEANBASE_PASSWORD,
        db_name=OCEANBASE_DB,
        table_name=RAG_TABLE_NAME,
        vec_column_name="embedding",
        limit=5,
        distance_func="cosine_distance",
        output_columns=["id", "text"],
    )
    return OceanBaseVectorSearchTool(
        oceanbase_config=config,
        custom_embedding_fn=get_embedding,
    )


def get_crew_response(query: str) -> str:
    """
    Run RAG: search OceanBase with query, then synthesize answer from results.
    Returns the final answer string from the RAG answer agent.
    """
    agents_config = _load_yaml("agents.yaml")
    tasks_config = _load_yaml("tasks.yaml")
    search_tool = _build_oceanbase_tool()

    # Build agents from YAML (semantic_search_agent, rag_answer_agent)
    semantic_search_agent = Agent(
        role=agents_config["semantic_search_agent"]["role"],
        goal=agents_config["semantic_search_agent"]["goal"],
        backstory=agents_config["semantic_search_agent"]["backstory"],
        llm=qwen_llm,
        tools=[search_tool],
        verbose=True,
    )
    rag_answer_agent = Agent(
        role=agents_config["rag_answer_agent"]["role"],
        goal=agents_config["rag_answer_agent"]["goal"],
        backstory=agents_config["rag_answer_agent"]["backstory"],
        llm=qwen_llm,
        verbose=True,
    )

    # Build tasks from YAML (rag_search_task, rag_answer_task); {query} is interpolated by kickoff(inputs=...)
    rag_search_task = Task(
        description=tasks_config["rag_search_task"]["description"],
        expected_output=tasks_config["rag_search_task"]["expected_output"],
        agent=semantic_search_agent,
    )
    rag_answer_task = Task(
        description=tasks_config["rag_answer_task"]["description"],
        expected_output=tasks_config["rag_answer_task"]["expected_output"],
        agent=rag_answer_agent,
        context=[rag_search_task],
    )

    crew = Crew(
        agents=[semantic_search_agent, rag_answer_agent],
        tasks=[rag_search_task, rag_answer_task],
        verbose=True,
    )
    result = crew.kickoff(inputs={"query": query})
    # CrewOutput: use final output (last task = rag_answer_task)
    if hasattr(result, "raw") and result.raw:
        return result.raw
    if hasattr(result, "tasks_output") and result.tasks_output:
        return result.tasks_output[-1] if result.tasks_output else str(result)
    return str(result)
