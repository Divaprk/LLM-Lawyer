"""
embed_and_store.py — Phase 2: Embed + Store
============================================

What this script does:
  1. Loads all three .jsonl chunk files
  2. Splits oversized chunks (SLA / eLitigation) with sliding window
     — Employment Act sections are already the right size, left intact
  3. Embeds every chunk using BAAI/bge-base-en-v1.5 (open-source, local)
  4. Stores vectors + metadata in ChromaDB (local, no server needed)
  5. Builds a BM25 keyword index (for hybrid search)
  6. Saves the BM25 index to disk so retrieval.py can load it

Requirements:
    pip install sentence-transformers chromadb rank_bm25

Run:
    python embed_and_store.py

Output:
    data/vectorstore/          ← ChromaDB persistent storage
    data/bm25_index.pkl        ← BM25 index + corpus
    data/processed_chunks.jsonl ← all final chunks after splitting (for inspection)
"""

import json
import re
import pickle
import time
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Input chunk files — update paths if yours differ
CHUNK_FILES = {
    "statute":   "data/chunks/employment_act.jsonl",
    "guideline": "data/chunks/singapore_legal_advice.jsonl",
    "case":      "data/chunks/elitigation_cases.jsonl",
}

# Output paths
VECTORSTORE_DIR     = Path("data/vectorstore")
BM25_INDEX_PATH     = Path("data/bm25_index.pkl")
PROCESSED_CHUNKS    = Path("data/processed_chunks.jsonl")
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# BGE-base embedding model
# Downloads ~440MB on first run, cached to ~/.cache/huggingface after that
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# ChromaDB collection name
COLLECTION_NAME = "sg_employment_law"

# Splitting config
# BGE-base has a 512 token limit. ~4 chars/token → 2048 chars max.
# We use 1500 chars to stay safely under, with 200 char overlap.
SPLIT_THRESHOLD = 1500   # chars — chunks longer than this get split
SPLIT_SIZE      = 1500   # chars per sub-chunk
SPLIT_OVERLAP   = 200    # chars of overlap between sub-chunks

# Embedding batch size — reduce to 16 if you run out of RAM
BATCH_SIZE = 32


# ─────────────────────────────────────────────
# STEP 1: Load all chunks
# ─────────────────────────────────────────────

def load_chunks() -> list[dict]:
    all_chunks = []
    for source_type, path in CHUNK_FILES.items():
        p = Path(path)
        if not p.exists():
            print(f"  [SKIP] {path} not found — skipping {source_type}")
            continue
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        chunks = [json.loads(l) for l in lines if l.strip()]
        print(f"  Loaded {len(chunks):>4} chunks from {path}")
        all_chunks.extend(chunks)

    print(f"\n  Total loaded: {len(all_chunks)} chunks")
    return all_chunks


# ─────────────────────────────────────────────
# STEP 2: Split oversized chunks
# ─────────────────────────────────────────────

def split_text_into_windows(text: str, size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping windows, breaking on sentence/paragraph
    boundaries where possible (never mid-sentence).

    Tries to split at: paragraph breaks → sentence endings → word boundaries
    """
    if len(text) <= size:
        return [text]

    # Prefer splitting at paragraph breaks, then sentence ends
    # This regex splits on double newlines OR after sentence-ending punctuation
    sentence_ends = re.compile(r"(?<=[\.\!\?])\s+|\n{2,}")
    sentences = sentence_ends.split(text)

    windows   = []
    current   = []
    cur_len   = 0

    for sent in sentences:
        sent_len = len(sent)

        # If even a single sentence is longer than size, hard-split it by words
        if sent_len > size:
            words    = sent.split()
            word_buf = []
            word_len = 0
            for w in words:
                if word_len + len(w) + 1 > size and word_buf:
                    windows.append(" ".join(word_buf))
                    # Keep overlap
                    keep = []
                    kept = 0
                    for wb in reversed(word_buf):
                        if kept + len(wb) < overlap:
                            keep.insert(0, wb)
                            kept += len(wb) + 1
                        else:
                            break
                    word_buf = keep + [w]
                    word_len = sum(len(x) + 1 for x in word_buf)
                else:
                    word_buf.append(w)
                    word_len += len(w) + 1
            if word_buf:
                current.append(" ".join(word_buf))
                cur_len += word_len
            continue

        if cur_len + sent_len > size and current:
            windows.append(" ".join(current))

            # Build overlap: keep tail of current buffer up to OVERLAP chars
            overlap_buf = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) < overlap:
                    overlap_buf.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break
            current = overlap_buf + [sent]
            cur_len = overlap_len + sent_len
        else:
            current.append(sent)
            cur_len += sent_len

    if current:
        windows.append(" ".join(current))

    return [w.strip() for w in windows if w.strip()]


def split_large_chunks(chunks: list[dict]) -> list[dict]:
    """
    For each chunk:
      - If text length <= SPLIT_THRESHOLD → keep as-is (Employment Act sections)
      - If text length >  SPLIT_THRESHOLD → split into overlapping sub-chunks

    Sub-chunks inherit all metadata from the parent.
    Their embed_text is rebuilt with the same metadata prefix but new sub-text.
    """
    result  = []
    n_split = 0

    for chunk in chunks:
        text = chunk["text"]

        if len(text) <= SPLIT_THRESHOLD:
            result.append(chunk)
            continue

        # Split the text body
        sub_texts = split_text_into_windows(text, SPLIT_SIZE, SPLIT_OVERLAP)
        n_split  += 1

        for i, sub_text in enumerate(sub_texts):
            # Build embed_text prefix depending on source type
            source_type = chunk.get("source_type", "")

            if source_type == "statute":
                prefix = (
                    f"Employment Act 1968, {chunk.get('part','')}, "
                    f"Section {chunk.get('section','')} — {chunk.get('section_title','')}:"
                )
            elif source_type == "guideline":
                prefix = f"Singapore Employment Law: {chunk.get('title', '')}"
            elif source_type == "case":
                prefix = (
                    f"SG Court Judgment: {chunk.get('case_name','')}\n"
                    f"Court: {chunk.get('court','')} | Year: {chunk.get('year','')} "
                    f"| Section: {chunk.get('section_heading', chunk.get('section_type',''))}"
                )
            else:
                prefix = chunk.get("title", chunk.get("case_name", ""))

            embed_text = f"{prefix}:\n\n{sub_text}"

            sub_chunk = {
                **chunk,                                # inherit all parent metadata
                "chunk_id":    f"{chunk['chunk_id']}_s{i}",
                "text":        sub_text,
                "embed_text":  embed_text,
                "is_sub_chunk":     True,
                "sub_chunk_index":  i,
                "sub_chunk_total":  len(sub_texts),
                "parent_chunk_id":  chunk["chunk_id"],
            }
            result.append(sub_chunk)

    n_kept = len(chunks) - n_split
    print(f"  Chunks kept as-is:   {n_kept}")
    print(f"  Chunks split:        {n_split}")
    print(f"  Total after split:   {len(result)}")
    return result


# ─────────────────────────────────────────────
# STEP 3: Embed with BGE
# ─────────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """
    Embed the embed_text of every chunk using BAAI/bge-base-en-v1.5.

    BGE models are trained with a query prefix for retrieval:
      - At QUERY time:    prepend "Represent this sentence: "
      - At DOCUMENT time: no prefix needed (what we do here)

    Returns a list of embedding vectors in the same order as chunks.
    """
    print(f"\n[EMBED] Loading model: {EMBEDDING_MODEL}")
    print("        (First run downloads ~440MB — cached after that)")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["embed_text"] for c in chunks]
    total = len(texts)
    print(f"  Embedding {total} chunks in batches of {BATCH_SIZE}...")

    all_embeddings = []
    t0 = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        # normalize_embeddings=True is recommended for BGE cosine similarity
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.extend(embeddings.tolist())

        # Progress
        done = min(i + BATCH_SIZE, total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done:>4}/{total}] {rate:.0f} chunks/sec | ETA {eta:.0f}s", end="\r")

    print(f"\n  Done in {time.time()-t0:.1f}s. Embedding dim: {len(all_embeddings[0])}")
    return all_embeddings


# ─────────────────────────────────────────────
# STEP 4: Store in ChromaDB
# ─────────────────────────────────────────────

def store_in_chroma(chunks: list[dict], embeddings: list[list[float]]):
    """
    Store all chunks + embeddings in a persistent ChromaDB collection.

    ChromaDB stores:
      - The vector (embedding)
      - The document text (for display)
      - Metadata fields (for filtering)

    Metadata fields we expose for filtering:
      source_type, section, part, act_name, case_name, year,
      section_type, is_repealed, url
    """
    print(f"\n[CHROMA] Storing in ChromaDB at {VECTORSTORE_DIR}...")

    import chromadb
    client     = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    # Delete existing collection if re-running
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        # Use cosine similarity (BGE embeddings are normalized so this is correct)
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB has a 41665 item limit per batch — use 500 to be safe
    CHROMA_BATCH = 500
    total = len(chunks)

    for i in range(0, total, CHROMA_BATCH):
        batch_chunks = chunks[i : i + CHROMA_BATCH]
        batch_embeds = embeddings[i : i + CHROMA_BATCH]

        ids        = [c["chunk_id"] for c in batch_chunks]
        documents  = [c["text"] for c in batch_chunks]
        metadatas  = [_build_metadata(c) for c in batch_chunks]

        collection.add(
            ids        = ids,
            embeddings = batch_embeds,
            documents  = documents,
            metadatas  = metadatas,
        )
        print(f"  Stored {min(i+CHROMA_BATCH, total)}/{total} chunks", end="\r")

    print(f"\n  Collection '{COLLECTION_NAME}' ready. Total: {collection.count()} vectors.")


def _build_metadata(chunk: dict) -> dict:
    """
    Extract a flat metadata dict from a chunk.
    ChromaDB metadata values must be str, int, float, or bool — no lists.
    """
    meta = {
        "source_type":    chunk.get("source_type", ""),
        "url":            chunk.get("url", ""),
        "scraped_at":     chunk.get("scraped_at", ""),
        "is_sub_chunk":   chunk.get("is_sub_chunk", False),
        "parent_chunk_id": chunk.get("parent_chunk_id", ""),
    }

    if chunk.get("source_type") == "statute":
        meta.update({
            "act_name":      chunk.get("act_name", ""),
            "part":          chunk.get("part", ""),
            "section":       chunk.get("section", ""),
            "section_title": chunk.get("section_title", ""),
            "is_repealed":   chunk.get("is_repealed", False),
            # cross_refs is a list — join to string for ChromaDB
            "cross_refs":    ", ".join(chunk.get("cross_refs", [])),
        })

    elif chunk.get("source_type") == "guideline":
        meta.update({
            "title":    chunk.get("title", ""),
            "category": chunk.get("category", ""),
        })

    elif chunk.get("source_type") == "case":
        meta.update({
            "case_name":       chunk.get("case_name", ""),
            "court":           chunk.get("court", ""),
            "year":            chunk.get("year", ""),
            "section_type":    chunk.get("section_type", ""),
            "section_heading": chunk.get("section_heading", ""),
        })

    return meta


# ─────────────────────────────────────────────
# STEP 5: Build BM25 keyword index
# ─────────────────────────────────────────────

def build_bm25_index(chunks: list[dict]):
    """
    Build a BM25 index over all chunk texts for keyword search.

    BM25 is essential for legal queries like:
      - "section 38"       → exact section number match
      - "workman"          → defined legal term
      - "retrenchment"     → specific legal concept
    These won't reliably surface via semantic search alone.

    Saves a dict containing:
      - 'bm25': the fitted BM25 object
      - 'chunk_ids': list of chunk_ids in the same order as BM25 corpus
      - 'corpus': list of tokenized texts (for reference)
    """
    print(f"\n[BM25] Building keyword index...")
    from rank_bm25 import BM25Okapi

    def tokenize(text: str) -> list[str]:
        """Simple tokenizer: lowercase, split on non-alphanumeric, remove stopwords."""
        STOPWORDS = {
            "the", "a", "an", "is", "in", "of", "to", "and", "or", "for",
            "on", "at", "by", "as", "be", "this", "that", "with", "from",
            "are", "was", "were", "has", "have", "had", "not", "no", "any",
            "it", "its", "he", "she", "they", "their", "which", "who"
        }
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    corpus     = [tokenize(c["text"]) for c in chunks]
    chunk_ids  = [c["chunk_id"] for c in chunks]

    bm25 = BM25Okapi(corpus)

    payload = {
        "bm25":       bm25,
        "chunk_ids":  chunk_ids,
        "corpus":     corpus,         # keep for debugging
        "built_at":   datetime.utcnow().isoformat(),
        "num_chunks": len(chunks),
    }

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"  BM25 index built over {len(chunks)} chunks.")
    print(f"  Saved to {BM25_INDEX_PATH}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 2 — Embed & Store")
    print("=" * 60)

    # ── 1. Load ──
    print("\n[LOAD] Reading chunk files...")
    chunks = load_chunks()
    if not chunks:
        print("[ERROR] No chunks loaded. Check CHUNK_FILES paths.")
        return

    # ── 2. Split ──
    print("\n[SPLIT] Splitting oversized chunks...")
    chunks = split_large_chunks(chunks)

    # ── Save processed chunks for inspection ──
    with open(PROCESSED_CHUNKS, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  All processed chunks saved to {PROCESSED_CHUNKS}")

    # ── 3. Embed ──
    embeddings = embed_chunks(chunks)

    # ── 4. Store in ChromaDB ──
    store_in_chroma(chunks, embeddings)

    # ── 5. BM25 index ──
    build_bm25_index(chunks)

    # ── Summary ──
    by_type = {}
    for c in chunks:
        t = c.get("source_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    print("\n" + "=" * 60)
    print("DONE — Phase 2 complete")
    print(f"  Total chunks embedded: {len(chunks)}")
    for t, n in sorted(by_type.items()):
        print(f"    {t:<12}: {n}")
    print(f"\n  ChromaDB:  {VECTORSTORE_DIR.resolve()}")
    print(f"  BM25:      {BM25_INDEX_PATH.resolve()}")
    print(f"  Chunks:    {PROCESSED_CHUNKS.resolve()}")
    print("\nNext step: run retrieval.py to test search.")
    print("=" * 60)


if __name__ == "__main__":
    main()
