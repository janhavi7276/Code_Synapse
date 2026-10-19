"""
file_tools.py — File Information & Onboarding Tools.
Lists project files, provides structural file summaries, and delivers onboarding setup guidance.
"""

import ast
from pathlib import Path


def _get_project_root() -> Path:
    """Resolve the root folder of the Code Synapse project."""
    return Path(__file__).resolve().parent.parent.parent


def _get_target_codebase_dir() -> Path:
    """Resolve the target Naive RAG directory (Naive_RAG/ or project root)."""
    root = _get_project_root()
    naive_rag = root / "Naive_RAG"
    if naive_rag.exists():
        return naive_rag
    return root


def list_project_files() -> str:
    """
    Lists all files in the project directory with their line count, size, and role.
    Use this tool when asked: 'What files exist in this project?', 'List all files',
    or to explore the project file structure.
    """
    target_dir = _get_target_codebase_dir()
    py_files = sorted(target_dir.glob("*.py"))

    descriptions = {
        "config.py": "Centralized configuration & API key management (NVIDIA NIM & Pinecone).",
        "indexer.py": "Offline indexing pipeline (Loads documents, chunks text, embeds via NIM, stores in Pinecone).",
        "vector_store.py": "Vector database interface for Pinecone Serverless and local JSON fallback.",
        "query_pipeline.py": "Online query pipeline (Embeds query, retrieves top-k chunks, prompts LLM).",
        "main.py": "Command-line demo runner orchestrating the full lecture walkthrough.",
    }

    dir_label = "Naive_RAG/" if target_dir.name == "Naive_RAG" else ""
    output = [f"=== PROJECT FILES IN NAIVE RAG REPOSITORY ({dir_label}) ==="]
    for pf in py_files:
        name = pf.name
        with open(pf, "r", encoding="utf-8") as f:
            lines = f.readlines()
        size_bytes = pf.stat().st_size
        desc = descriptions.get(name, "Python module.")
        output.append(f"\n* {dir_label}{name} ({len(lines)} lines, {size_bytes} bytes)")
        output.append(f"  Role: {desc}")

    # Check for data directory
    data_dir = target_dir / "data"
    if data_dir.exists():
        output.append(f"\nKnowledge Documents ({dir_label}data/):")
        for df in sorted(data_dir.glob("*.txt")):
            output.append(f"  - {dir_label}data/{df.name} ({df.stat().st_size} bytes)")

    return "\n".join(output)


def get_file_summary(file_name: str) -> str:
    """
    Extracts high-level structure of a specific file: line count, imports, classes, and functions.
    Use this tool to prepare file information for explanation or when asked:
    'Give me a summary of file X', or to inspect classes/functions in a specific file.
    """
    target_dir = _get_target_codebase_dir()
    clean_name = file_name.strip()
    if clean_name.startswith("Naive_RAG/") or clean_name.startswith("Naive_RAG\\"):
        clean_name = clean_name.replace("Naive_RAG/", "").replace("Naive_RAG\\", "")
    if not clean_name.endswith(".py"):
        clean_name += ".py"
    target_path = target_dir / clean_name
    if not target_path.exists():
        target_path = _get_project_root() / clean_name

    if not target_path.exists():
        return f"I could not find that. File '{file_name}' does not exist in the project."

    with open(target_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        content = "".join(lines)

    tree = ast.parse(content, filename=clean_name)
    docstring = ast.get_docstring(tree) or "No module docstring."

    functions = []
    classes = []
    imports = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fn_doc = ast.get_docstring(node)
            summary = fn_doc.strip().splitlines()[0] if fn_doc else "No docstring"
            functions.append(f"{node.name}() [Line {node.lineno}] - {summary}")
        elif isinstance(node, ast.ClassDef):
            classes.append(f"class {node.name} [Line {node.lineno}]")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))

    output = [
        f"=== FILE SUMMARY: {clean_name} ===",
        f"Total Lines: {len(lines)} lines",
        f"Module Docstring: {docstring.strip().splitlines()[0]}",
        f"\nImports ({len(imports)}):",
    ]
    output.extend([f"  - {imp}" for imp in imports] if imports else ["  - None"])
    output.append(f"\nClasses ({len(classes)}):")
    output.extend([f"  - {cls}" for cls in classes] if classes else ["  - None"])
    output.append(f"\nFunctions ({len(functions)}):")
    output.extend([f"  - {fn}" for fn in functions] if functions else ["  - None"])
    return "\n".join(output)


def get_project_quickstart() -> str:
    """
    Returns exact prerequisites, setup commands, and execution instructions for the project.
    Use this tool when a user asks: 'I am new, where do I start?', 'How do I run the project?',
    or asks for setup steps and dependencies.
    """
    return """=== NAIVE RAG PROJECT QUICKSTART & SETUP GUIDE ===

1. PREREQUISITES:
   - Python 3.10+ (tested on Python 3.13)
   - Node.js (v18+) for Filesystem MCP tools
   - NVIDIA NIM API Key (for LLM and 2048-dim embeddings)
   - Pinecone API Key (free serverless account)

2. INSTALLATION & ENVIRONMENT SETUP (Windows PowerShell):
   cd C:\\Naive_RAG
   python -m venv .venv
   .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
   .\\.venv\\Scripts\\python.exe -m pip install google-adk "litellm>=1.84" mcp pinecone python-dotenv

3. CONFIGURE CREDENTIALS (.env):
   Create a .env file in the project root with:
   NVIDIA_API_KEY=nvapi-your-key-here
   NVIDIA_NIM_API_KEY=nvapi-your-key-here
   PINECONE_API_KEY=your-pinecone-key
   PINECONE_INDEX_NAME=code-synapse
   PINECONE_CLOUD=aws
   PINECONE_REGION=us-east-1

4. RUNNING THE NAIVE RAG DEMO (main.py):
   - Full End-to-End Walkthrough:  python main.py
   - Run Offline Indexing Only:    python main.py --index
   - Run Custom Question:          python main.py --query "What is the employee hardware stipend?"
   - Run Baseline vs. RAG:         python main.py --compare

5. RUNNING MULTI-AGENT CODE SYNAPSE (ADK):
   - Index the codebase:           python code_synapse/index_codebase.py
   - Launch Web UI (Recommended):  adk web
   - Launch Terminal CLI:          adk run code_synapse

"""


def get_project_architecture_overview() -> str:
    """
    Returns the conceptual 8-stage architecture of the Naive RAG pipeline.
    Use this tool when a user asks what the project does, how Naive RAG works,
    or how the 8 pipeline stages are distributed across the files.
    """
    return """=== 8-STAGE NAIVE RAG ARCHITECTURE OVERVIEW ===

The project is structured into two sequential pipelines across 8 explicit stages:

OFFLINE INDEXING PIPELINE (indexer.py + vector_store.py):
  [Stage 1] Document Ingestion   : load_document() reads source text into memory.
  [Stage 2] Chunking             : chunk_document() uses a sliding character window (500 chars, 100 overlap).
  [Stage 3] Embedding Generation : embed_chunks() generates 2048-dim vectors via NVIDIA NIM (nvidia/nemotron-3-embed-1b).
  [Stage 4] Vector DB Storage    : store_in_vector_db() upserts embeddings + text metadata into Pinecone Serverless.

ONLINE QUERY PIPELINE (query_pipeline.py + vector_store.py):
  [Stage 5] Query Embedding      : embed_query() embeds user question into the identical 2048-dim space.
  [Stage 6] Retrieval            : retrieve_top_k() performs cosine similarity nearest-neighbor search.
  [Stage 7] Context Injection    : build_prompt_with_context() injects retrieved chunks into an augmented prompt.
  [Stage 8] Grounded Generation  : generate_answer() calls NVIDIA NIM chat completion with low temperature (0.1).

FILE ROLES AT A GLANCE:
  - config.py         : Loads environment variables, validates credentials, sets hyperparameters.
  - query_pipeline.py : Executes Stages 5-8 (query embedding, retrieval, prompt assembly, and answer generation).
  - main.py           : Command-line orchestrator that coordinates the full lecture walkthrough.
"""


# Backward-compatible alias
get_file_outline = get_file_summary
