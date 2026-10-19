"""
architecture_tools.py — Static Analysis Tools for Code Architecture & Inspection.
Uses Python's native AST parser (Pure Python, NO AI) to deterministically
map file imports, dependencies, and function definitions.
"""

import ast
from pathlib import Path


def _get_project_root() -> Path:
    """Resolve the root folder of the Code Synapse repository."""
    return Path(__file__).resolve().parent.parent.parent


def _get_target_codebase_dir() -> Path:
    """Resolve the target Naive RAG directory (Naive_RAG/ or project root)."""
    root = _get_project_root()
    naive_rag = root / "Naive_RAG"
    if naive_rag.exists():
        return naive_rag
    return root


def map_imports(file_name: str = "") -> str:
    """
    Inspects Python file imports and inter-file dependencies using AST parsing.
    Use this tool when asked about file connections, which files import config.py,
    or the overall dependency graph of the project without AI guessing.
    """
    root = _get_target_codebase_dir()
    py_files = sorted(root.glob("*.py"))

    if not py_files:
        return "No Python source files found in the project root."

    file_names = {f.name for f in py_files}

    # If a specific file is queried
    if file_name and file_name.strip():
        target = file_name.strip()
        if not target.endswith(".py"):
            target += ".py"
        target_path = root / target
        if not target_path.exists():
            return f"I could not find that. File '{file_name}' does not exist in the project."

        # Which files import this target file
        importers = []
        for f_path in py_files:
            if f_path.name == target:
                continue
            with open(f_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mod = getattr(node, "module", None) or ""
                    names = [n.name for n in getattr(node, "names", [])]
                    target_base = target.replace(".py", "")
                    if target_base == mod or target_base in names:
                        importers.append(f_path.name)

        importers_unique = sorted(set(importers))
        result = [f"=== IMPORT ANALYSIS FOR: {target} ==="]
        if importers_unique:
            result.append(f"* {target} is imported by: {', '.join(importers_unique)}")
        else:
            result.append(f"* {target} is not imported by any other project file (Entry point or standalone).")
        return "\n".join(result)

    # General project-wide dependency graph
    results = ["=== PROJECT IMPORT & DEPENDENCY GRAPH (AST PARSED, NO AI) ==="]

    for f_path in py_files:
        rel_name = f_path.name
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content, filename=rel_name)
        except Exception as e:
            results.append(f"\n[{rel_name}] - Error parsing file: {e}")
            continue

        local_imports = []
        external_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    target_file = f"{module_name}.py"
                    if target_file in file_names and target_file != rel_name:
                        local_imports.append(target_file)
                    else:
                        external_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    target_file = f"{module_name}.py"
                    if target_file in file_names and target_file != rel_name:
                        local_imports.append(target_file)
                    else:
                        external_imports.append(node.module)

        local_unique = sorted(set(local_imports))
        external_unique = sorted(set(external_imports))

        results.append(f"\nFile: {rel_name}")
        if local_unique:
            results.append(f"  -> Imports Project Files : {', '.join(local_unique)}")
        else:
            results.append("  -> Imports Project Files : (None - Standalone module)")

        if external_unique:
            results.append(f"  -> External Dependencies : {', '.join(external_unique)}")

    # Reverse lookup: Which files import each file
    results.append("\n--- REVERSE DEPENDENCY LOOKUP (Who imports what?) ---")
    for target in sorted(file_names):
        importers = []
        for f_path in py_files:
            if f_path.name == target:
                continue
            with open(f_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mod = getattr(node, "module", None) or ""
                    names = [n.name for n in getattr(node, "names", [])]
                    target_base = target.replace(".py", "")
                    if target_base == mod or target_base in names:
                        importers.append(f_path.name)

        importers_unique = sorted(set(importers))
        if importers_unique:
            results.append(f"  * {target} is imported by: {', '.join(importers_unique)}")
        else:
            results.append(f"  * {target} is not imported by any other project file (Entry point or standalone).")

    return "\n".join(results)


def find_function(function_name: str = "") -> str:
    """
    Locates function definitions, signatures, line numbers, and docstrings using Python AST.
    Use this tool when asked: 'Where is function X defined?', 'Find function X',
    or to list all function definitions across the codebase without AI guessing.
    """
    root = _get_target_codebase_dir()
    py_files = sorted(root.glob("*.py"))
    clean_query = function_name.strip().lower()

    matches = []
    all_functions = []

    for f_path in py_files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=f_path.name)
        except Exception as err:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                docstring = ast.get_docstring(node)
                first_line = docstring.strip().split("\n")[0] if docstring else "No docstring"
                sig = f"def {node.name}({', '.join(args)})"
                entry = {
                    "file": f_path.name,
                    "name": node.name,
                    "line": node.lineno,
                    "signature": sig,
                    "summary": first_line,
                }
                all_functions.append(entry)
                if clean_query and clean_query in node.name.lower():
                    matches.append(entry)

    if clean_query:
        if not matches:
            return f"I could not find that. No function matching '{function_name}' was found in the project files."
        output = [f"=== FOUND {len(matches)} FUNCTION(S) MATCHING: '{function_name}' ==="]
        for m in matches:
            output.append(f"\n* Function : {m['signature']}")
            output.append(f"  Location : {m['file']} (Line {m['line']})")
            output.append(f"  Summary  : {m['summary']}")
        return "\n".join(output)

    # If no query provided, list all functions
    output = ["=== ALL FUNCTION DEFINITIONS ACROSS PROJECT (AST PARSED) ==="]
    current_file = ""
    for m in all_functions:
        if m["file"] != current_file:
            current_file = m["file"]
            output.append(f"\n--- {current_file} ---")
        output.append(f"  * Line {m['line']:>3}: {m['signature']}")
        output.append(f"    Summary: {m['summary']}")

    return "\n".join(output)


# Backward-compatible aliases
find_file_imports_and_dependencies = map_imports
inspect_function_definitions = find_function
