# Course Experiments

Jupyter notebook with day-by-day experiments following the [AI Hero](https://aishippinglabs.com/courses/aihero) crash course curriculum.

## Contents

`experiments.ipynb` covers:

- **Day 1 - Ingestion**: Download and parse markdown files from GitHub repos
- **Day 2 - Chunking**: Intelligent (LLM-based) and simple (sliding window) chunking strategies
- **Day 3 - Search**: Text search, vector search, and hybrid search with minsearch
- **Day 4 - Agents**: Pydantic AI agent with search tool, system prompt iterations (v1/v2)
- **Day 5 - Evaluation**: Interaction logging, log simplification, LLM-as-judge scoring
- **Day 6 - Deployment**: Code extraction into modular Python scripts (see `project/src/`)
- **Day 7 - Documentation**: README, demo, and sharing

## Setup

```bash
# From the repository root
uv sync

# Activate the virtual environment (Bash / Git Bash on Windows)
source .venv/Scripts/activate

# On Unix/macOS
# source .venv/bin/activate

# Configure API key (the notebook loads from course/.env, the project from project/.env)
echo "OPENAI_API_KEY=your-key-here" > course/.env
```

Then open `experiments.ipynb` in Jupyter or VS Code.

## Note

The production code lives in [`project/src/`](../project/src/). This notebook is for experimentation and learning.
