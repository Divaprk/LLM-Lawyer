# Singapore Employment Law Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about Singapore employment law, grounded in Singapore statutes, MOM guidelines, Tripartite advisories, and court judgments. Powered by OpenAI (`gpt-4o-mini`) and a locally-built vector index.

## Demo

Ask questions like:
- *"My boss fired me without notice after 3 years. What are my rights?"*
- *"How many days of annual leave am I entitled to?"*
- *"My salary hasn't been paid for 2 months. What can I do?"*
- *"What are my rights under the Workplace Fairness Act?"*
- *"Can my employer deny my flexible work arrangement request?"*

## Features

- **Hybrid search** — BM25 keyword + semantic vector search fused via Reciprocal Rank Fusion (RRF); catches both exact legal terms and paraphrased questions
- **Weighted retrieval** — source-type multipliers (statute / guideline / case) applied post-RRF to calibrate result mix against corpus composition
- **Topic-boosted retrieval** — query matched against domain keywords (workplace safety, FWA, PDPA, etc.) to sharpen search terms before retrieval
- **LLM query rewriting** — vague questions converted to precise legal terms before retrieval
- **Structure-aware chunking** — statutes chunked by section; PDFs chunked by detected headings (font size or ALL-CAPS patterns); preserves legal document hierarchy
- **Parent-document retrieval** — matched sub-chunks escalated to full article/section context for richer answers
- **Citation verification** — hallucinated act/section citations flagged automatically after generation
- **WFA grace period notices** — programmatic disclaimer appended whenever Workplace Fairness Act sections are cited
- **Conversation memory** — last 6 turns kept in context so follow-up questions work naturally

## Knowledge Base

| Source | Chunks | Content |
|--------|--------|---------|
| Employment Act 1968 | 154 | All 154 enacted sections with full statutory text |
| CPF Act | 154 | CPF contribution and withdrawal obligations |
| Child Development Co-Savings Act | 51 | Maternity/paternity leave provisions |
| Work Injury Compensation Act | 111 | Workplace injury claims and compensation |
| Retirement and Re-employment Act | 23 | Retirement age and re-employment obligations |
| Workplace Safety and Health Act | 73 | Workplace safety duties and offences |
| Personal Data Protection Act | 86 | Employee data handling obligations |
| Employment of Foreign Manpower Act | 51 | Foreign worker pass conditions |
| Workplace Fairness Act | 46 | Anti-discrimination provisions (grace period until 2027) |
| Industrial Relations Act | 109 | Union and collective bargaining rights |
| TAFEP Tripartite Standards | 9 | Fair employment practice guidelines |
| FWA Tripartite Guidelines | 15 | Flexible work arrangement request/response rules |
| MOM WorkRight Guide | 8 | MOM plain-English guide to employment rights |
| SingaporeLegalAdvice.com | 66 | Plain-English employment law articles |
| eLitigation court judgments | 97 | ECT and High Court employment cases |
| **Total** | **1,053** | |

**Sources:**
- https://sso.agc.gov.sg (all statutes)
- https://www.tal.sg/tafep/employment-practices/tripartite-standards
- https://www.mom.gov.sg (WorkRight Guide, FWA Guidelines)
- https://singaporelegaladvice.com/law-articles/employment-law/
- https://www.elitigation.sg

---

## Setup

### Prerequisites

- Python 3.10 or higher
- OpenAI API key
- ~2GB free disk space (vector index)

### Step 1 — Clone the repo

```bash
git clone <repo-url>
cd LLM-Lawyer\test1
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

### Step 4 — Set your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

Or export it in your shell:

```bash
# Windows
set OPENAI_API_KEY=sk-...

# Mac/Linux
export OPENAI_API_KEY=sk-...
```

### Step 5 — Build the vector store

This embeds all chunks and builds the search indexes. **Run once only.**

```bash
python embed_and_store.py
```

Expected output:
```
Total loaded: 1053 chunks
Total after split: 4392
statute: 1670 | guideline: 809 | case: 1913
Embedding 4392 chunks in batches of 32...
Collection 'sg_employment_law' ready. Total: 4392 vectors.
BM25 index built over 4392 chunks.
```

This creates two items that are **not** in the repo (too large for git):
- `data/vectorstore/` — ChromaDB vector index
- `data/bm25_index.pkl` — BM25 keyword index

### Step 6 — Launch the chatbot

```bash
streamlit run chatbot.py
```

Your browser will open automatically at `http://localhost:8501`.
