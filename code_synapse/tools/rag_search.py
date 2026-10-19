"""
rag_search.py — Codebase RAG Search Tool.
Searches the indexed Python codebase using NVIDIA NIM embeddings and Pinecone.
"""

from pathlib import Path
import sys

# Ensure code_synapse directory is in python path
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from index_codebase import search_codebase


def search_code(query: str) -> str:
    """
    Searches the indexed Python codebase for functions, logic, variables, and comments.
    Use this tool when asked: 'Where is something implemented?', 'Where is the search done?',
    or to find exact lines of code across the 5 project files.
    """
    clean_query = query.strip()
    if not clean_query:
        return "I could not find that. Please provide a search query."

    try:
        matches = search_codebase(query=clean_query, top_k=4)
    except Exception as err:
        return f"Error executing RAG search: {err}"

    if not matches:
        return "I could not find that in the indexed codebase. Please verify if the code exists or run index_codebase.py."

    output = [
        f"=== CODEBASE RAG SEARCH RESULTS FOR: '{clean_query}' ===",
        f"Found {len(matches)} relevant code passages:\n",
    ]

    for idx, m in enumerate(matches, 1):
        score_pct = round(m["score"] * 100, 1)
        snippet = m["text"].strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "\n... [truncated]"
        output.append(f"--- [Match #{idx}] File: {m['file']} (Relevance: {score_pct}%, Chunk: {m['id']}) ---")
        output.append(snippet)
        output.append("")

    return "\n".join(output)


# Backward-compatible alias
rag_search = search_code
