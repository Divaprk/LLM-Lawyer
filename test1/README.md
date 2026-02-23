# Singapore Employment Law Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about Singapore employment law, grounded in the Employment Act, MOM guidelines, and court judgments. Runs fully locally.

## Demo

Ask questions like:
- *"My boss fired me without notice after 3 years. What are my rights?"*
- *"How many days of annual leave am I entitled to?"*
- *"My salary hasn't been paid for 2 months. What can I do?"*
- *"What does section 38 say about hours of work?"*

## Features

- **Hybrid search** — BM25 keyword + semantic vector search combined
- **LLM query rewriting** — vague questions converted to precise legal terms before retrieval
- **Parent-document retrieval** — matched sub-chunks escalated to full article context
- **Citation verification** — hallucinated citations flagged automatically
- **Conversation memory** — follow-up questions work naturally within a session

## Knowledge Base

| Source | Chunks | Content |
|--------|--------|---------|
| Singapore Employment Act 1968 | 154 | All 146 live sections with full text |
| SingaporeLegalAdvice.com | 66 | Plain-English employment law articles |
| eLitigation court judgments | 97 | ECT and High Court employment cases |

- https://singaporelegaladvice.com/law-articles/employment-law/

- https://www.elitigation.sg/gd/Home/Index?Filter=SUPCT&YearOfDecision=All&SortBy=Score&SearchPhrase=employment&CurrentPage=5&SortAscending=False&PageSize=0&Verbose=False&SearchQueryTime=0&SearchTotalHits=0&SearchMode=False&SpanMultiplePages=False

- https://sso.agc.gov.sg/act/ema1968#al-

---

## Setup

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com) (free, local LLM runner)
- ~6GB free disk space (model + vector index)

### Step 1 — Clone the repo

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install Ollama and download the LLM

1. Go to [https://ollama.com](https://ollama.com) and download the installer for your OS
2. Install it (it runs as a background service automatically)
3. Open a terminal and pull the model:

```bash
ollama pull llama3.1:8b
```

This downloads ~5GB once and caches it. Subsequent runs use the cache.

> **Smaller/faster alternatives** if you are low on disk space:
> ```bash
> ollama pull llama3.2      # 2GB — good for most questions
> ollama pull mistral       # 4GB — better reasoning
> ```
> If you use a different model, update `LLM_MODEL` at the top of `chatbot.py`.

### Step 5 — Build the vector store

This embeds all chunks and builds the search indexes. **Run once only** — takes about 10 minutes on CPU, faster with a GPU.

```bash
python embed_and_store.py
```

Expected output:
```
Total loaded: 317 chunks
Total after split: 2886
Embedding 2886 chunks in batches of 32...
[2886/2886] Done in ~540s
Collection 'sg_employment_law' ready. Total: 2886 vectors.
BM25 index built over 2886 chunks.
```

This creates two items that are **not** in the repo (too large for git):
- `data/vectorstore/` — ChromaDB vector index
- `data/bm25_index.pkl` — BM25 keyword index

### Step 6 — Launch the chatbot

```bash
streamlit run chatbot.py
```

Your browser will open automatically at `http://localhost:8501`.
