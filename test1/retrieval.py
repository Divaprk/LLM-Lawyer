"""
retrieval.py — Hybrid Search (BM25 + Vector)
============================================

Can be run standalone to test your search pipeline:
    python retrieval.py

Or imported by your chatbot:
    from retrieval import retrieve
    results = retrieve("can my boss fire me without notice?")

HOW HYBRID SEARCH WORKS
------------------------
1. BM25 (keyword):  scores all chunks by exact term overlap
                    → catches "section 38", "workman", legal jargon
2. Vector (semantic): scores all chunks by meaning similarity
                    → catches paraphrases, vague questions
3. Reciprocal Rank Fusion (RRF): merges the two ranked lists
                    → top results = relevant by BOTH meaning AND keywords

The query is also rewritten before retrieval to make it more legal-specific.
E.g. "my boss fired me" → "wrongful dismissal without notice Employment Act"
"""

import re
import pickle
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

VECTORSTORE_DIR  = Path("data/vectorstore")
BM25_INDEX_PATH  = Path("data/bm25_index.pkl")
COLLECTION_NAME  = "sg_employment_law"
EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"

# How many results to fetch from each search before merging
BM25_TOP_K   = 20
VECTOR_TOP_K = 20

# How many final results to return after merging
FINAL_TOP_K  = 5

# RRF constant (60 is standard — don't change unless you have a reason)
RRF_K = 60

# Source-type score multipliers applied after RRF merge.
# Statutes are boosted because they make up only ~8% of the corpus
# (226 of 2886 chunks) and are chronically under-retrieved without help.
SOURCE_WEIGHTS = {
    "statute":   3.0,   # Employment Act sections — heavily underrepresented
    "guideline": 1.0,   # SingaporeLegalAdvice — baseline
    "case":      1.0,   # eLitigation — 66% of corpus, no boost needed
}

# BGE query prefix — required for BGE models at query time
# (document embeddings don't use a prefix, but queries must)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ─────────────────────────────────────────────
# LOAD INDEXES (done once, cached in memory)
# ─────────────────────────────────────────────

_bm25_payload  = None
_chroma_client = None
_collection    = None
_embed_model   = None


def _load_bm25():
    global _bm25_payload
    if _bm25_payload is None:
        if not BM25_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {BM25_INDEX_PATH}. "
                "Run embed_and_store.py first."
            )
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25_payload = pickle.load(f)
        print(f"[BM25] Loaded index: {_bm25_payload['num_chunks']} chunks")
    return _bm25_payload


def _load_chroma():
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        _collection    = _chroma_client.get_collection(COLLECTION_NAME)
        print(f"[CHROMA] Loaded collection '{COLLECTION_NAME}': {_collection.count()} vectors")
    return _collection


def _load_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[EMBED] Model loaded: {EMBEDDING_MODEL}")
    return _embed_model


# ─────────────────────────────────────────────
# QUERY REWRITING
# ─────────────────────────────────────────────

# Simple rule-based rewriter — works without an LLM call.
# For Phase 3 you can upgrade this to an actual LLM rewrite.

TOPIC_KEYWORDS = {
    "wrongful dismissal": [
        "fired", "sacked", "dismissed", "terminate", "termination",
        "wrongful", "unfair dismissal", "let go", "fired without reason"
    ],
    "notice period": [
        "notice", "resign", "resignation", "how much notice", "serving notice"
    ],
    "salary payment": [
        "unpaid salary", "salary not paid", "late salary", "wages", "pay me",
        "owe me money", "withhold pay", "deduct salary"
    ],
    "annual leave": [
        "leave entitlement", "annual leave", "days off", "time off", "vacation"
    ],
    "maternity leave": [
        "maternity", "pregnant", "pregnancy", "confinement", "childcare leave"
    ],
    "retrenchment": [
        "retrenched", "retrenchment", "redundancy", "laid off", "layoff"
    ],
    "CPF": [
        "cpf", "central provident fund", "cpf contributions", "employer cpf"
    ],
    "overtime": [
        "overtime", "extra hours", "work more than 8 hours", "work on rest day"
    ],
    "sick leave": [
        "sick leave", "medical leave", "mc", "hospitalisation leave", "unwell"
    ],
    "employment contract": [
        "contract", "agreement", "terms of employment", "offer letter", "bond"
    ],
    "workplace discrimination": [
        "discriminat", "unfair treatment", "biased", "race", "religion", "gender"
    ],
    "work injury": [
        "injured at work", "workplace accident", "wica", "compensation for injury"
    ],
}


def rewrite_query(query: str) -> str:
    """
    Expand the user's query with legal terminology.
    This is a rule-based pre-filter — replace with an LLM call in Phase 3.

    E.g. "my boss fired me without telling me" 
      → "wrongful dismissal without notice Singapore Employment Act remedies"
    """
    q_lower = query.lower()
    matched_topics = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            matched_topics.append(topic)

    if not matched_topics:
        # No specific topic detected — append generic legal context
        return f"{query} Singapore employment law rights obligations"

    legal_context = " ".join(matched_topics)
    return f"{query} {legal_context} Singapore Employment Act"


# ─────────────────────────────────────────────
# SEARCH FUNCTIONS
# ─────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    STOPWORDS = {
        "the", "a", "an", "is", "in", "of", "to", "and", "or", "for",
        "on", "at", "by", "as", "be", "this", "that", "with", "from",
        "are", "was", "were", "has", "have", "had", "not", "no", "any",
        "it", "its", "he", "she", "they", "their", "which", "who"
    }
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def bm25_search(query: str, top_k: int = BM25_TOP_K) -> list[dict]:
    """Return top_k chunk_ids and BM25 scores for the query."""
    payload = _load_bm25()
    bm25    = payload["bm25"]
    ids     = payload["chunk_ids"]

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)

    # Get indices of top_k scores
    import heapq
    top_indices = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])

    return [
        {"chunk_id": ids[i], "bm25_score": float(scores[i]), "bm25_rank": rank + 1}
        for rank, i in enumerate(top_indices)
        if scores[i] > 0  # skip zero-score results
    ]


def vector_search(query: str, top_k: int = VECTOR_TOP_K,
                  filter_metadata: dict = None) -> list[dict]:
    """
    Return top_k results from ChromaDB vector search.

    filter_metadata: optional ChromaDB where clause for pre-filtering
    e.g. {"source_type": "statute"} to search only Employment Act chunks
    """
    model      = _load_embed_model()
    collection = _load_chroma()

    # BGE requires this prefix on the QUERY (not on documents)
    prefixed_query = BGE_QUERY_PREFIX + query
    query_embedding = model.encode(
        prefixed_query,
        normalize_embeddings=True,
    ).tolist()

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if filter_metadata:
        kwargs["where"] = filter_metadata

    results = collection.query(**kwargs)

    output = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        # ChromaDB cosine distance → similarity (lower distance = more similar)
        similarity = 1 - dist
        output.append({
            "chunk_id":       results["ids"][0][i],
            "text":           doc,
            "metadata":       meta,
            "vector_score":   similarity,
            "vector_rank":    i + 1,
        })

    return output


# ─────────────────────────────────────────────
# HYBRID MERGE: Reciprocal Rank Fusion
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score = 1/(k + rank_bm25) + 1/(k + rank_vector)

    A chunk that ranks #1 in both gets the highest possible combined score.
    A chunk that appears in only one list still gets partial credit.
    """
    scores = {}  # chunk_id → rrf_score

    for item in bm25_results:
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + item["bm25_rank"])

    for item in vector_results:
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + item["vector_rank"])

    # Build a lookup for full result data
    full_data = {}
    for item in bm25_results + vector_results:
        cid = item["chunk_id"]
        if cid not in full_data:
            full_data[cid] = item

    # Sort by RRF score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**full_data[cid], "rrf_score": score}
        for cid, score in ranked
    ]


# ─────────────────────────────────────────────
# PUBLIC API: retrieve()
# ─────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = FINAL_TOP_K,
    rewrite: bool = True,
    filter_source: str = None,
    verbose: bool = False,
) -> list[dict]:
    """
    Main retrieval function. Call this from your chatbot.

    Args:
        query:         The user's question (natural language)
        top_k:         Number of results to return (default 5)
        rewrite:       Whether to apply query rewriting (default True)
        filter_source: Optionally restrict to one source type:
                       "statute" | "guideline" | "case"
        verbose:       Print debug info

    Returns:
        List of dicts, each containing:
          chunk_id, text, metadata, rrf_score, vector_score, bm25_score
    """
    # ── Query rewriting ──
    effective_query = rewrite_query(query) if rewrite else query

    if verbose:
        print(f"\nOriginal query:  {query}")
        print(f"Rewritten query: {effective_query}")

    # ── Build metadata filter ──
    meta_filter = {"source_type": filter_source} if filter_source else None

    # ── Run both searches ──
    bm25_results   = bm25_search(effective_query, top_k=BM25_TOP_K)
    vector_results = vector_search(effective_query, top_k=VECTOR_TOP_K,
                                   filter_metadata=meta_filter)

    if verbose:
        print(f"\nBM25 top-3:   {[r['chunk_id'] for r in bm25_results[:3]]}")
        print(f"Vector top-3: {[r['chunk_id'] for r in vector_results[:3]]}")

    # ── Merge ──
    merged = reciprocal_rank_fusion(bm25_results, vector_results)

    # ── Fetch full metadata for BM25-only results (BM25 doesn't return metadata) ──
    collection = _load_chroma()
    chunk_ids_need_meta = [
        r["chunk_id"] for r in merged[:top_k]
        if "metadata" not in r
    ]
    if chunk_ids_need_meta:
        fetched = collection.get(ids=chunk_ids_need_meta,
                                 include=["documents", "metadatas"])
        meta_lookup = {
            cid: {"text": doc, "metadata": meta}
            for cid, doc, meta in zip(
                fetched["ids"],
                fetched["documents"],
                fetched["metadatas"],
            )
        }
        for r in merged:
            if r["chunk_id"] in meta_lookup:
                r.update(meta_lookup[r["chunk_id"]])

    # ── Apply source-type weights and re-sort ──
    for r in merged:
        src = r.get("metadata", {}).get("source_type", "")
        r["rrf_score"] = r["rrf_score"] * SOURCE_WEIGHTS.get(src, 1.0)
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)

    return merged[:top_k]


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────

TEST_QUERIES = [
    "Can my employer fire me without giving a reason?",
    "What are my annual leave entitlements?",
    "My salary has not been paid for 2 months, what can I do?",
    "What does section 38 say about hours of work?",
    "I was retrenched, am I entitled to retrenchment benefit?",
]

def run_test():
    print("=" * 65)
    print("Retrieval Test — Hybrid BM25 + Vector Search")
    print("=" * 65)

    for query in TEST_QUERIES:
        print(f"\nQ: {query}")
        print("-" * 65)

        results = retrieve(query, top_k=3, verbose=False)

        for i, r in enumerate(results, 1):
            meta    = r.get("metadata", {})
            src     = meta.get("source_type", "?")
            rrf     = r.get("rrf_score", 0)

            # Build a source label
            if src == "statute":
                label = f"Employment Act s.{meta.get('section','')} — {meta.get('section_title','')}"
            elif src == "guideline":
                label = f"SLA: {meta.get('title','')[:50]}"
            elif src == "case":
                label = f"Case: {meta.get('case_name','')[:50]}"
            else:
                label = r.get("chunk_id", "")

            text_preview = r.get("text", "")[:150].replace("\n", " ")

            print(f"  [{i}] {label}")
            print(f"       Score: {rrf:.4f} | {text_preview}...")

    print("\n" + "=" * 65)
    print("Test complete. If results look relevant, retrieval is working.")
    print("Next step: build the LLM layer (chatbot.py).")
    print("=" * 65)


if __name__ == "__main__":
    run_test()
