"""
code_synapse.tools package.
Exports all custom tools matching Phase 3 specifications and aliases.
"""

from .rag_search import search_code, rag_search
from .file_tools import (
    list_project_files,
    get_file_summary,
    get_file_outline,
    get_project_quickstart,
    get_project_architecture_overview,
)
from .architecture_tools import (
    map_imports,
    find_function,
    find_file_imports_and_dependencies,
    inspect_function_definitions,
)

__all__ = [
    "search_code",
    "rag_search",
    "list_project_files",
    "get_file_summary",
    "get_file_outline",
    "get_project_quickstart",
    "get_project_architecture_overview",
    "map_imports",
    "find_function",
    "find_file_imports_and_dependencies",
    "inspect_function_definitions",
]
