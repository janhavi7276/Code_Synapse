"""
agent.py — The Multi-Agent Code Synapse Team.
Implements 1 Boss (coordinator) and 4 specialized workers using Google ADK & LiteLLM.
Follows Phases 4, 5, and 6 of the implementation plan.
"""

import os
from pathlib import Path
import shutil
from dotenv import load_dotenv

# Ensure UTF-8 encoding on Windows to prevent UnicodeDecodeError (Student Guide Page 4)
os.environ["PYTHONUTF8"] = "1"

# 1. Load credentials from .env in code_synapse or project root
agent_dir = Path(__file__).resolve().parent
project_root = agent_dir.parent

if (agent_dir / ".env").exists():
    load_dotenv(agent_dir / ".env", override=True)

if (project_root / ".env").exists():
    load_dotenv(project_root / ".env", override=True)
load_dotenv()

# 2. ADK model initialization
from google.adk.agents import LlmAgent
from google.adk.models import LiteLlm, Gemini
from google.genai import types
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from .tools import (
    search_code,
    list_project_files,
    get_file_summary,
    map_imports,
    find_function,
    get_project_quickstart,
    get_project_architecture_overview,
)

# Configure NVIDIA NIM API via LiteLLM
nvidia_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY")
if nvidia_key:
    os.environ["NVIDIA_API_KEY"] = nvidia_key
    os.environ["NVIDIA_NIM_API_KEY"] = nvidia_key
    chat_model_id = os.getenv("NVIDIA_CHAT_MODEL", "meta/llama-3.2-11b-vision-instruct")
    model = LiteLlm(model=f"nvidia_nim/{chat_model_id}")
else:
    gemini_model_id = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model = Gemini(model=gemini_model_id)

# Token limits to prevent verbose loops and keep responses snappy
ROUTER_CONFIG = types.GenerateContentConfig(temperature=0.0, max_output_tokens=256)
WORKER_CONFIG = types.GenerateContentConfig(temperature=0.2, max_output_tokens=512)


# =====================================================================
# PHASE 6: CONNECT FILESYSTEM MCP SERVER (CONNECTED TO CODE EXPLAINER)
# =====================================================================
# Resolves npx executable path on Windows (e.g. C:\Program Files\nodejs\npx.CMD)
npx_binary = shutil.which("npx") or "npx"
file_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=npx_binary,
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                str(project_root),
            ],
        ),
        timeout=30,
    ),
    tool_filter=["list_directory", "read_file"],
)


# =====================================================================
# PHASE 4: WORKER 1 — CODE SEARCHER (Where is it?)
# =====================================================================
searcher = LlmAgent(
    name="code_searcher",
    model=model,
    description="Finds where functions, variables, or logic are in the code.",
    instruction=(
        "You are code_searcher. NEVER call transfer_to_agent. "
        "Call the search_code tool at most ONCE. Do NOT call search_code multiple times. "
        "Immediately answer the user directly and concisely based strictly on the search results."
    ),
    tools=[search_code],
    generate_content_config=WORKER_CONFIG,
)


# =====================================================================
# PHASE 5: WORKER 2 — CODE EXPLAINER (What does this file do?)
# =====================================================================
explainer = LlmAgent(
    name="code_explainer",
    model=model,
    description="Reads a whole file and explains it using the filesystem MCP server.",
    instruction=(
        "You are code_explainer. NEVER call transfer_to_agent. "
        "Use the filesystem tools (read_file) or get_file_summary to inspect the requested file, "
        "and then explain its purpose and functionality clearly and concisely."
    ),
    tools=[file_tools, get_file_summary],
    generate_content_config=WORKER_CONFIG,
)


# =====================================================================
# PHASE 5: WORKER 3 — ARCHITECTURE MAPPER (How are the files connected?)
# =====================================================================
mapper = LlmAgent(
    name="architecture_mapper",
    model=model,
    description="Lists files, functions and imports with plain Python AST. No AI guessing.",
    instruction=(
        "You are architecture_mapper. NEVER call transfer_to_agent. "
        "Use map_imports or find_function to inspect the structure of files in this codebase, "
        "and answer strictly based on the tool output."
    ),
    tools=[map_imports, find_function],
    generate_content_config=WORKER_CONFIG,
)


# =====================================================================
# PHASE 5: WORKER 4 — ONBOARDING GUIDE (Where do I start?)
# =====================================================================
guide = LlmAgent(
    name="onboarding_guide",
    model=model,
    description="Says what the project does, how to run it, and where to start.",
    instruction=(
        "You are onboarding_guide. NEVER call transfer_to_agent. "
        "Help new developers understand the project, how to set it up, how to run it, "
        "and where to start using list_project_files, get_project_quickstart, or get_project_architecture_overview."
    ),
    tools=[list_project_files, get_project_quickstart, get_project_architecture_overview],
    generate_content_config=WORKER_CONFIG,
)


# =====================================================================
# PHASE 5: BOSS / COORDINATOR (Routes queries, NEVER answers code itself)
# =====================================================================
root_agent = LlmAgent(
    name="coordinator",
    model=model,
    description="Sends each question to the right worker.",
    instruction=(
        "You are the coordinator for the Code Synapse multi-agent team. "
        "For greetings such as 'hi' or 'hello', reply politely in one short sentence: "
        "'Hi! I am the Code Synapse coordinator. How can I help you explore this codebase?' Do NOT transfer greetings. "
        "For all questions about the code, do NOT answer yourself. Transfer to the best specialist worker: "
        "- code_searcher: finds where functions, variables, or logic are in the code. "
        "- code_explainer: reads and explains whole files using the filesystem tools. "
        "- architecture_mapper: analyzes imports and function definitions with AST. "
        "- onboarding_guide: explains what the project does, architecture overview, and how to run it."
    ),
    sub_agents=[searcher, explainer, mapper, guide],
    generate_content_config=ROUTER_CONFIG,
)
