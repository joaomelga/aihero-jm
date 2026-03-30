"""
RAG agent with search tool and interaction logging.
"""

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, List

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter

from src.config import AGENT_MODEL, LOG_DIR, load_env
from src.search import vector_search
 
# todo: make implementation less specific for Evidently or add the option to the user to configure the system prompt, model name, etc
SYSTEM_PROMPT = """
You are a helpful assistant about Evidently - an open-source Python library to evaluate, test, and monitor ML and LLM systems, from experiments to production.

Use the search tool to find relevant information before answering questions.

If you can find specific information through search, use it to provide accurate answers.

Always include references by citing the filename of the source material you used.

When citing the reference, replace "evidently-main" by the full path to the GitHub repository: "https://github.com/evidentlyai/evidently/blob/main/"
Format: [LINK TITLE](FULL_GITHUB_LINK)

If the search doesn't return relevant results, let the user know and provide general guidance.
""".strip()


def text_search(query: str) -> List[Any]:
    """
    Perform a vector search on the knowledge base.

    Args:
        query: The search query string.

    Returns:
        A list of up to 5 search results from the knowledge base.
    """
    return vector_search(query, num_results=5)


def create_agent() -> Agent:
    """Create and return the RAG agent."""
    load_env()
    return Agent(
        name="evidently_agent",
        instructions=SYSTEM_PROMPT,
        tools=[text_search],
        model=AGENT_MODEL,
    )


def _serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def log_entry(agent: Agent, messages, source: str = "user") -> dict:
    """Build a log entry dict from agent and messages."""
    tools = []
    for ts in agent.toolsets:
        tools.extend(ts.tools.keys())

    dict_messages = ModelMessagesTypeAdapter.dump_python(messages)

    return {
        "agent_name": agent.name,
        "system_prompt": agent._instructions,
        "provider": agent.model.system,
        "model": agent.model.model_name,
        "tools": tools,
        "messages": dict_messages,
        "source": source,
    }


def log_interaction_to_file(
    agent: Agent, messages, source: str = "user"
) -> Path:
    """Log an agent interaction to a JSON file. Returns the file path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    entry = log_entry(agent, messages, source)
    ts = entry["messages"][-1]["timestamp"]

    if not isinstance(ts, datetime):
        ts = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))

    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    rand_hex = secrets.token_hex(3)

    filename = f"{agent.name}_{ts_str}_{rand_hex}.json"
    filepath = LOG_DIR / filename

    with filepath.open("w", encoding="utf-8") as f_out:
        json.dump(entry, f_out, indent=2, default=_serializer)

    return filepath
