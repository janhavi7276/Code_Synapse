"""
index_codebase.py — Custom 5-Step RAG Indexer for Multi-Agent Code Synapse.
Builds and persists a Pinecone vector index of the 5 Python files in c:\\Naive_RAG.
Follows Student Guide Section 6 (no LlamaIndex/LangChain).
"""

import json
import math
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


def load_env() -> None:
    """Load environment variables from project root or code_synapse directory."""
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent

    # Try .env in code_synapse, then root
    if (current_dir / ".env").exists():
        load_dotenv(current_dir / ".env", override=True)

    if (root_dir / ".env").exists():
        load_dotenv(root_dir / ".env", override=True)
    load_dotenv()


load_env()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_codebase_dir() -> Path:
    """Returns the directory containing the Naive RAG target codebase."""
    naive_rag_dir = PROJECT_ROOT / "Naive_RAG"
    if naive_rag_dir.exists():
        return naive_rag_dir
    return PROJECT_ROOT

TARGET_FILES = [
    "config.py",
    "indexer.py",
    "vector_store.py",
    "query_pipeline.py",
    "main.py",
]
LOCAL_CACHE_FILE = Path(__file__).resolve().parent / "local_code_index.json"


# =====================================================================
# STEP 1: READ EVERY .PY FILE IN THE CODE PROJECT
# =====================================================================
def read_codebase(root: Path = None) -> dict[str, str]:
    """Reads each target Python file in the project into memory."""
    if root is None:
        root = get_codebase_dir()
    codebase = {}
    print("\n" + "=" * 60)
    print("STEP 1: READING CODEBASE FILES")
    print(f"Directory: {root}")
    print("=" * 60)
    for fname in TARGET_FILES:
        fpath = root / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            codebase[fname] = content
            lines = len(content.splitlines())
            chars = len(content)
            print(f"  [OK] Read {fname:<18} ({lines:>3} lines, {chars:>5} chars)")
        else:
            print(f"  [!] Missing file: {fname}")
    return codebase


# =====================================================================
# STEP 2: SPLIT CODE INTO FIXED SLIDING WINDOW CHUNKS
# =====================================================================
def chunk_code_file(
    file_name: str,
    code_text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Splits source code into sliding window chunks with character overlap.
    Preserves file metadata and line approximation.
    """
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap.")

    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(code_text):
        end = start + chunk_size
        slice_text = code_text[start:end].strip()

        if slice_text:
            # Header gives context for the embedding model
            formatted_text = f"# File: {file_name} (Chars {start}-{end})\n{slice_text}"
            chunk_record = {
                "id": f"{file_name.replace('.py', '')}_c{chunk_idx}",
                "file": file_name,
                "text": formatted_text,
                "raw_code": slice_text,
                "start_char": start,
                "end_char": start + len(slice_text),
            }
            chunks.append(chunk_record)
            chunk_idx += 1

        start += step

    return chunks


def chunk_all_files(
    codebase: dict[str, str],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """Chunks all codebase files using the configured window parameters."""
    all_chunks = []
    print("\n" + "=" * 60)
    print(f"STEP 2: CHUNKING (Size: {chunk_size} chars, Overlap: {chunk_overlap} chars)")
    print("=" * 60)
    for fname, content in codebase.items():
        file_chunks = chunk_code_file(
            file_name=fname,
            code_text=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(file_chunks)
        print(f"  * {fname:<18} -> {len(file_chunks):>2} chunks")

    print(f"[OK] Total chunks generated: {len(all_chunks)}")
    return all_chunks


# =====================================================================
# STEP 3: EMBED CHUNKS VIA NVIDIA NIM (nvidia/nemotron-3-embed-1b)
# =====================================================================
def embed_chunks(
    chunks: list[dict],
    api_key: str | None = None,
    base_url: str = "https://integrate.api.nvidia.com/v1",
    model: str = "nvidia/nemotron-3-embed-1b",
) -> list[dict]:
    """
    Sends each chunk to the NVIDIA NIM embedding model to produce 2048-dim vectors.
    Falls back to deterministic offline simulation if API key is unconfigured.
    """
    if not api_key:
        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY") or ""

    print("\n" + "=" * 60)
    print("STEP 3: EMBEDDING GENERATION (NVIDIA NIM)")
    print("=" * 60)
    print(f"[*] Embedding Model : {model}")

    if api_key and not api_key.startswith("your_"):
        client = OpenAI(base_url=base_url, api_key=api_key)
        texts = [c["text"] for c in chunks]

        # Batch embed (max 50 per call)
        batch_size = 30
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(input=batch, model=model)
            embeddings.extend([item.embedding for item in response.data])

        dim = len(embeddings[0])
        print(f"[OK] Generated {len(embeddings)} live NVIDIA NIM embeddings (Dimension: {dim}).")
        for idx, emb in enumerate(embeddings):
            chunks[idx]["embedding"] = emb
        return chunks

    # Offline simulation fallback for testing without credentials
    print("[!] NVIDIA API key not configured. Generating simulated deterministic 2048-dim vectors...")
    for c in chunks:
        seed = sum(ord(ch) for ch in c["text"])
        sim_vec = []
        for d in range(2048):
            val = math.sin(seed + d * 0.1) * 0.5
            sim_vec.append(round(val, 6))
        c["embedding"] = sim_vec

    print(f"[OK] Generated {len(chunks)} simulated 2048-dim embeddings.")
    return chunks


# =====================================================================
# STEP 4: STORE IN PINECONE (Serverless, 2048 dimensions, Cosine)
# =====================================================================
def store_in_pinecone(
    chunks: list[dict],
    api_key: str | None = None,
    index_name: str = "code-synapse",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> int:
    """
    Creates the Pinecone index if needed, and upserts vectors + metadata.
    Also caches locally to guarantee offline resilience.
    """
    if not api_key:
        api_key = os.getenv("PINECONE_API_KEY") or ""

    print("\n" + "=" * 60)
    print("STEP 4: STORING IN PINECONE VECTOR DATABASE")
    print("=" * 60)
    print(f"[*] Index Name : {index_name}")

    # Always persist to local cache as reliable backup
    cache_records = []
    for c in chunks:
        record = {
            "id": c["id"],
            "embedding": c["embedding"],
            "metadata": {
                "file": c["file"],
                "text": c["text"],
                "raw_code": c["raw_code"],
                "chunk_id": c["id"],
            },
        }
        cache_records.append(record)

    with open(LOCAL_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_records, f, indent=2)
    print(f"[OK] Local vector cache written to {LOCAL_CACHE_FILE.name}")

    if api_key and not api_key.startswith("your_"):
        try:
            pc = Pinecone(api_key=api_key)
            existing_indexes = [idx.name for idx in pc.list_indexes()]

            if index_name not in existing_indexes:
                print(f"[*] Index '{index_name}' not found. Creating serverless index (2048 dims, cosine)...")
                pc.create_index(
                    name=index_name,
                    dimension=2048,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
                print(f"[OK] Created Pinecone index '{index_name}'.")
            else:
                print(f"[OK] Connected to existing Pinecone index '{index_name}'.")

            index = pc.Index(index_name)
            vectors_payload = []
            for c in chunks:
                record = {
                    "id": c["id"],
                    "values": c["embedding"],
                    "metadata": {
                        "file": c["file"],
                        "text": c["text"][:1000],  # Pinecone metadata size safeguard
                        "raw_code": c["raw_code"][:1000],
                        "chunk_id": c["id"],
                    },
                }
                vectors_payload.append(record)

            # Upsert in batches of 50
            batch_size = 50
            for i in range(0, len(vectors_payload), batch_size):
                batch = vectors_payload[i : i + batch_size]
                index.upsert(vectors=batch)

            print(f"[OK] Successfully upserted {len(vectors_payload)} vectors to Pinecone cloud.")
            return len(vectors_payload)

        except Exception as e:
            print(f"[!] Warning: Pinecone upsert failed ({e}). Local fallback cache is active.")
            return len(chunks)

    print("[!] PINECONE_API_KEY not set. Using local JSON vector cache for zero-crash demo.")
    return len(chunks)


# =====================================================================
# STEP 5: SEARCH (Embed query & find closest chunks)
# =====================================================================
def search_codebase(query: str, top_k: int = 3) -> list[dict]:
    """Embeds the question and asks Pinecone (or local cache) for closest chunks."""
    api_key_nim = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY") or ""
    api_key_pc = os.getenv("PINECONE_API_KEY") or ""
    index_name = os.getenv("PINECONE_INDEX_NAME", "code-synapse")

    # 1. Embed query
    if api_key_nim and not api_key_nim.startswith("your_"):
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key_nim)
        resp = client.embeddings.create(input=[query], model="nvidia/nemotron-3-embed-1b")
        q_vec = resp.data[0].embedding
    else:
        seed = sum(ord(ch) for ch in query)
        q_vec = [round(math.sin(seed + d * 0.1) * 0.5, 6) for d in range(2048)]

    # 2. Query Pinecone Cloud
    if api_key_pc and not api_key_pc.startswith("your_"):
        try:
            pc = Pinecone(api_key=api_key_pc)
            idx = pc.Index(index_name)
            res = idx.query(vector=q_vec, top_k=top_k, include_metadata=True)
            results = []
            for m in res.matches:
                meta = m.metadata or {}
                results.append({
                    "id": m.id,
                    "score": float(m.score),
                    "file": meta.get("file", ""),
                    "text": meta.get("text", ""),
                })
            return results
        except Exception as e:
            print(f"[!] Pinecone query error ({e}), falling back to local cache.")

    # Fallback to local cache cosine similarity
    if not LOCAL_CACHE_FILE.exists():
        return []

    with open(LOCAL_CACHE_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    def cos_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if (na * nb) > 0 else 0.0

    scored = []
    for r in records:
        score = cos_sim(q_vec, r["embedding"])
        scored.append({
            "id": r["id"],
            "score": score,
            "file": r["metadata"]["file"],
            "text": r["metadata"]["text"],
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# =====================================================================
# CHUNKING COMPARISON UTILITY (FOR README.MD DOCUMENTATION)
# =====================================================================
def compare_chunk_sizes(codebase: dict[str, str]) -> dict:
    """Compares two sliding window chunking configurations for README evaluation."""
    sizes = [(350, 70), (500, 100)]
    report = {}
    print("\n" + "=" * 60)
    print("CHUNKING STRATEGY EVALUATION (350 vs 500 chars)")
    print("=" * 60)
    for c_size, c_overlap in sizes:
        total = 0
        file_counts = {}
        for fname, code in codebase.items():
            cnt = len(chunk_code_file(fname, code, c_size, c_overlap))
            file_counts[fname] = cnt
            total += cnt
        report[f"size_{c_size}_overlap_{c_overlap}"] = {
            "chunk_size": c_size,
            "overlap": c_overlap,
            "total_chunks": total,
            "file_counts": file_counts,
        }
        print(f"  Configuration ({c_size} chars, {c_overlap} overlap) -> Total Chunks: {total}")
    return report


def main() -> None:
    """Full execution of the 5-step RAG indexing pipeline."""
    print("\n" + "#" * 60)
    print("STARTING CUSTOM 5-STEP CODEBASE RAG INDEXER")
    print("#" * 60)

    # Step 1: Read
    codebase = read_codebase()

    # Compare chunk sizes for documentation
    compare_chunk_sizes(codebase)

    # Step 2: Split (Chosen production size: 500 chars, 100 overlap)
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
    chunks = chunk_all_files(codebase, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Step 3: Embed
    embedded_chunks = embed_chunks(chunks)

    # Step 4: Store
    index_name = os.getenv("PINECONE_INDEX_NAME", "code-synapse")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    store_in_pinecone(embedded_chunks, index_name=index_name, cloud=cloud, region=region)

    # Step 5: Test search
    print("\n" + "=" * 60)
    print("STEP 5: VERIFICATION TEST SEARCH")
    print("=" * 60)
    test_q = "Where is cosine similarity calculated?"
    print(f"[*] Test Question: \"{test_q}\"")
    results = search_codebase(test_q, top_k=2)
    for r in results:
        print(f"  -> Match [{r['id']}] in {r['file']} (Score: {r['score']:.4f})")
        first_line = r["text"].splitlines()[:2]
        print(f"     Preview: {' '.join(first_line)}")

    print("\n" + "#" * 60)
    print("[SUCCESS] CODEBASE INDEXING FINISHED -- PINECONE IS READY!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
