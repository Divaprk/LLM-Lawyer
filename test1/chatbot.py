"""
chatbot.py — Singapore Employment Law RAG Chatbot
==================================================

Features implemented:
  1. LLM Query Rewriting    — rewrites vague questions into legal search terms
  2. Hybrid Retrieval       — BM25 + vector search via retrieval.py
  3. Parent-Doc Retrieval   — sub-chunks escalate to full parent article/section
  4. Citation Verification  — flags hallucinated citations post-generation
  5. Conversation Memory    — last N turns kept in context for follow-up questions

LLM provider: OpenAI (gpt-4o-mini — cheap, fast, good at instruction following)
Switch to a different provider by changing the _call_llm() function.

Requirements:
    pip install openai streamlit

Usage:
    # Set your API key first (once):
    # Windows:  set OPENAI_API_KEY=sk-...
    # Mac/Linux: export OPENAI_API_KEY=sk-...

    streamlit run chatbot.py          # Web UI
    python chatbot.py                 # CLI test mode (no UI)
"""

import os
import re
import json
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# LLM settings
LLM_MODEL        = "gpt-4o-mini"
LLM_TEMPERATURE  = 0.1     # Low temp for factual legal answers
LLM_MAX_TOKENS   = 1024

# Retrieval settings
TOP_K_RETRIEVE   = 5       # chunks sent to LLM context
MEMORY_TURNS     = 6       # number of past turns to keep (3 user + 3 assistant)

# Parent-doc retrieval: max chars to include from parent chunk
PARENT_MAX_CHARS = 1500

# Processed chunks file (built by embed_and_store.py) — used for parent lookup
PROCESSED_CHUNKS_PATH = Path("data/processed_chunks.jsonl")


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Singapore Employment Law assistant. You help members of the public understand their rights and obligations under Singapore employment law.

RULES YOU MUST FOLLOW:
1. Base your answer ONLY on the retrieved context provided. Do not use outside knowledge.
2. Always cite your source. Use this exact format:
   - For Employment Act sections: [Employment Act s.{number}]
   - For SLA articles: [SingaporeLegalAdvice: {article title}]
   - For court cases: [Case: {case name}]
3. If the retrieved context does not contain enough information to answer the question, say exactly: "I'm sorry, I don't have enough information in my knowledge base to answer this question. Please consult a lawyer or visit mom.gov.sg for official guidance."
4. Never give a definitive legal ruling. Use language like "under the Employment Act...", "according to...", "you may be entitled to...".
5. Always end your response with this disclaimer: "⚠️ This is general information only and does not constitute legal advice. For your specific situation, please consult a qualified employment lawyer or the Ministry of Manpower (mom.gov.sg)."

FORMAT:
- Use plain, simple English that a non-lawyer can understand.
- Keep answers concise — 2 to 4 short paragraphs.
- If multiple sections or sources apply, address each briefly.
- Do not include section numbers in your prose — keep citations in brackets at the end of the relevant sentence."""


# ─────────────────────────────────────────────
# LLM CALLER
# ─────────────────────────────────────────────

def _call_llm(messages: list[dict], max_tokens: int = LLM_MAX_TOKENS) -> str:
    import ollama
    response = ollama.chat(
        model="llama3.1:8b",
        messages=messages,
        options={
            "temperature": LLM_TEMPERATURE,
            "num_predict": max_tokens,
        }
    )
    return response["message"]["content"].strip()


# ─────────────────────────────────────────────
# FEATURE 1: LLM Query Rewriting
# ─────────────────────────────────────────────

REWRITE_PROMPT = """You are a legal search assistant for Singapore employment law.

Your job: rewrite the user's question into a precise legal search query.

Rules:
- Replace casual language with legal terms (e.g. "fired" → "dismissal", "days off" → "annual leave entitlement")
- Add relevant Singapore legal context: act names, section topics, tribunal names (ECT, MOM)
- Keep it to 1-2 sentences maximum
- Do NOT answer the question — only rewrite it as a search query

Examples:
User: "my boss fired me out of nowhere"
Rewritten: "wrongful dismissal without notice Employment Act Singapore remedies ECT"

User: "how many days of leave do I get"
Rewritten: "annual leave entitlement days Employment Act Singapore employee rights"

User: "they haven't paid me for months"
Rewritten: "unpaid salary wages recovery Employment Act Singapore employee rights MOM"

Now rewrite this query. Reply with ONLY the rewritten query, nothing else:
"""


def rewrite_query_llm(query: str, conversation_context: str = "") -> str:
    """
    Use the LLM to rewrite the user's query into precise legal search terms.
    Falls back to the original query if the LLM call fails.
    """
    prompt = REWRITE_PROMPT
    if conversation_context:
        prompt += f"\nConversation context: {conversation_context}\n"
    prompt += f"\nUser query: {query}"

    try:
        rewritten = _call_llm([{"role": "user", "content": prompt}], max_tokens=100)
        # Sanity check: if rewritten is much longer than original, something went wrong
        if len(rewritten) > len(query) * 4:
            return query
        return rewritten
    except Exception as e:
        print(f"  [WARN] Query rewrite failed: {e}. Using original query.")
        return query


# ─────────────────────────────────────────────
# FEATURE 3: Parent-Document Retrieval
# ─────────────────────────────────────────────

# Build a parent chunk lookup once at startup (lazy-loaded)
_parent_lookup = None

def _load_parent_lookup() -> dict:
    """
    Load all original (pre-split) chunks into memory, keyed by chunk_id.
    Used to retrieve the full parent text when a sub-chunk matches.
    """
    global _parent_lookup
    if _parent_lookup is not None:
        return _parent_lookup

    _parent_lookup = {}
    if not PROCESSED_CHUNKS_PATH.exists():
        return _parent_lookup

    for line in PROCESSED_CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        _parent_lookup[chunk["chunk_id"]] = chunk

    return _parent_lookup


def escalate_to_parent(results: list[dict]) -> list[dict]:
    """
    For sub-chunks (is_sub_chunk=True), augment the result with
    context from the parent chunk.

    Employment Act sections are never split so they have no parent.
    SLA articles and eLitigation chunks that were split do have parents.

    We DON'T replace the matched sub-chunk — we ADD the parent title
    and surrounding context to it, giving the LLM more to work with.
    """
    lookup = _load_parent_lookup()

    for result in results:
        meta = result.get("metadata", {})
        if not meta.get("is_sub_chunk"):
            continue

        parent_id = meta.get("parent_chunk_id", "")
        if not parent_id or parent_id not in lookup:
            continue

        parent = lookup[parent_id]
        parent_text = parent.get("text", "")

        # Prepend parent context (title + opening) to the sub-chunk text
        if meta.get("source_type") == "guideline":
            parent_title   = parent.get("title", "")
            parent_context = f"[From article: {parent_title}]\n{parent_text[:PARENT_MAX_CHARS]}..."
        elif meta.get("source_type") == "case":
            case_name      = meta.get("case_name", "")
            parent_context = f"[From case: {case_name}]\n{parent_text[:PARENT_MAX_CHARS]}..."
        else:
            parent_context = parent_text[:PARENT_MAX_CHARS]

        # Augment — prepend parent context, keep the specific matched text
        result["text"]           = parent_context + "\n\n---\n" + result.get("text", "")
        result["parent_fetched"] = True

    return results


# ─────────────────────────────────────────────
# FEATURE 4: Citation Verification
# ─────────────────────────────────────────────

def verify_citations(answer: str, retrieved_chunks: list[dict]) -> tuple[str, list[str]]:
    """
    Check that every citation in the LLM's answer actually comes from
    a retrieved chunk. Flag any that don't.

    Returns:
        (verified_answer, list_of_warnings)

    Citation formats we check:
        [Employment Act s.38]     → check if any chunk is EA s.38
        [Case: Some v Other]      → check if any chunk is that case
        [SingaporeLegalAdvice: X] → check if any chunk is that SLA article
    """
    warnings = []

    # ── Extract all citations from the answer ──
    ea_citations  = re.findall(r"\[Employment Act s\.(\w+)\]", answer)
    case_citations = re.findall(r"\[Case:\s*([^\]]+)\]", answer)
    sla_citations  = re.findall(r"\[SingaporeLegalAdvice:\s*([^\]]+)\]", answer)

    # ── Build sets of what's actually in retrieved chunks ──
    retrieved_ea_sections = set()
    retrieved_cases       = set()
    retrieved_sla_titles  = set()

    for r in retrieved_chunks:
        meta = r.get("metadata", {})
        src  = meta.get("source_type", "")

        if src == "statute":
            retrieved_ea_sections.add(meta.get("section", "").lower())

        elif src == "case":
            case_name = meta.get("case_name", "").lower()
            # Add partial matches (case names can be truncated in citations)
            retrieved_cases.add(case_name)

        elif src == "guideline":
            title = meta.get("title", "").lower()
            retrieved_sla_titles.add(title)

    # ── Verify each citation ──
    for sec in ea_citations:
        if sec.lower() not in retrieved_ea_sections:
            warnings.append(
                f"⚠️ Citation [Employment Act s.{sec}] could not be verified "
                f"against retrieved sources."
            )

    for case in case_citations:
        case_lower = case.lower().strip()
        # Partial match — a citation might say "Smile Inc v Lui" for a long case name
        found = any(case_lower in rc or rc in case_lower for rc in retrieved_cases)
        if not found:
            warnings.append(
                f"⚠️ Citation [Case: {case}] could not be verified "
                f"against retrieved sources."
            )

    for title in sla_citations:
        title_lower = title.lower().strip()
        found = any(title_lower in rt or rt in title_lower for rt in retrieved_sla_titles)
        if not found:
            warnings.append(
                f"⚠️ Citation [SingaporeLegalAdvice: {title}] could not be "
                f"verified against retrieved sources."
            )

    return answer, warnings


# ─────────────────────────────────────────────
# FEATURE 5: Conversation Memory
# ─────────────────────────────────────────────

class ConversationMemory:
    """
    Keeps the last MEMORY_TURNS messages in context.

    Why this matters for employment law:
    A user asks "Was my termination valid?" then follows up with
    "Can I get compensation?". Without memory the second question
    has no context. With memory the LLM knows what was already discussed.

    We also maintain a running "user context" string — key facts the
    user has mentioned (salary, employment duration, job type) that
    should be prepended to every retrieval query.
    """

    def __init__(self, max_turns: int = MEMORY_TURNS):
        self.max_turns    = max_turns
        self.history: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]
        self.user_context = ""         # extracted facts about the user's situation

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        # Keep only the last max_turns messages
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_messages(self) -> list[dict]:
        """Return history in OpenAI message format."""
        return self.history.copy()

    def get_context_string(self) -> str:
        """Return the last 2 turns as plain text for query rewriting context."""
        recent = self.history[-4:] if len(self.history) >= 4 else self.history
        return " | ".join(f"{m['role']}: {m['content'][:100]}" for m in recent)

    def extract_user_context(self, user_message: str):
        """
        Extract key legal facts from what the user says about themselves.
        These facts get prepended to subsequent queries.

        E.g. "I earn $3,500/month" → stored and prepended to future searches
        """
        facts = []

        # Salary mentions
        salary = re.search(r"\$[\d,]+(?:\s*(?:per month|/month|monthly|a month))?", user_message, re.IGNORECASE)
        if salary:
            facts.append(f"salary: {salary.group(0)}")

        # Employment duration
        duration = re.search(r"(\d+)\s*(year|month)s?\s*(of service|employed|working)", user_message, re.IGNORECASE)
        if duration:
            facts.append(f"employment duration: {duration.group(0)}")

        # Job type keywords
        if re.search(r"\bworkman\b|\bblue.?collar\b|\bmanual\b", user_message, re.IGNORECASE):
            facts.append("job type: workman")
        elif re.search(r"\bmanager\b|\bexecutive\b|\bdirector\b", user_message, re.IGNORECASE):
            facts.append("job type: managerial/executive")

        if facts:
            # Append new facts, avoid duplicates
            new_context = "; ".join(facts)
            if new_context not in self.user_context:
                self.user_context = (self.user_context + "; " + new_context).strip("; ")

    def clear(self):
        self.history      = []
        self.user_context = ""


# ─────────────────────────────────────────────
# CORE: build_prompt() and answer()
# ─────────────────────────────────────────────

def build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.
    Each chunk is labelled with its source so the LLM can cite correctly.
    """
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        src  = meta.get("source_type", "")
        text = chunk.get("text", "")[:2000]  # cap per chunk to control prompt length

        if src == "statute":
            label = (f"[SOURCE {i}] Employment Act s.{meta.get('section','')} "
                     f"— {meta.get('section_title','')}")
        elif src == "guideline":
            label = f"[SOURCE {i}] SingaporeLegalAdvice: {meta.get('title','')}"
        elif src == "case":
            label = (f"[SOURCE {i}] Case: {meta.get('case_name','')} "
                     f"({meta.get('court','')}, {meta.get('year','')})")
        else:
            label = f"[SOURCE {i}]"

        blocks.append(f"{label}\n{text}")

    return "\n\n---\n\n".join(blocks)


def answer(
    query: str,
    memory: ConversationMemory,
    verbose: bool = False,
) -> tuple[str, list[str], list[dict]]:
    """
    Full RAG pipeline for one user turn.

    Returns:
        (answer_text, citation_warnings, retrieved_chunks)
    """
    from retrieval import retrieve

    # ── 1. Extract user context facts ──
    memory.extract_user_context(query)

    # ── 2. LLM query rewriting ──
    conv_context = memory.get_context_string()
    effective_query = rewrite_query_llm(query, conv_context)
    if verbose:
        print(f"  [REWRITE] {query!r}\n        → {effective_query!r}")

    # Append known user context to improve retrieval
    if memory.user_context:
        effective_query += f" {memory.user_context}"

    # ── 3. Hybrid retrieval ──
    raw_results = retrieve(effective_query, top_k=TOP_K_RETRIEVE, rewrite=False)

    # ── 4. Parent-document escalation ──
    results = escalate_to_parent(raw_results)

    if verbose:
        print(f"  [RETRIEVE] Top results:")
        for r in results:
            m = r.get("metadata", {})
            print(f"    {m.get('source_type','')} | s.{m.get('section','')} {m.get('title',m.get('case_name',''))[:40]} | score={r.get('rrf_score',0):.4f}")

    # ── 5. Build prompt ──
    context_block = build_context_block(results)
    user_content  = f"""User question: {query}

Retrieved context:
{context_block}

Answer the question based ONLY on the context above. Cite each source used."""

    # Add user context if available
    if memory.user_context:
        user_content = f"User background: {memory.user_context}\n\n" + user_content

    # Build full message list: system + history + current question
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + memory.get_messages()
        + [{"role": "user", "content": user_content}]
    )

    # ── 6. LLM generation ──
    llm_answer = _call_llm(messages)

    # ── 7. Citation verification ──
    verified_answer, warnings = verify_citations(llm_answer, results)

    # ── 8. Update conversation memory ──
    memory.add_turn("user", query)          # store original query (not rewritten)
    memory.add_turn("assistant", llm_answer)

    return verified_answer, warnings, results


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

def run_streamlit():
    import streamlit as st

    st.set_page_config(
        page_title="Singapore Employment Law Assistant",
        page_icon="⚖️",
        layout="wide",
    )

    # ── Session state ──
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of (role, content, warnings)

    # ── Sidebar ──
    with st.sidebar:
        st.title("⚖️ SG Employment Law")
        st.markdown("""
**Ask questions like:**
- *My boss fired me without reason. What can I do?*
- *How many days of annual leave am I entitled to?*
- *My salary hasn't been paid for 2 months*
- *Can my employer cut my pay without telling me?*
- *What is the notice period for resignation?*
        """)
        st.divider()
        if st.button("🗑️ Clear conversation"):
            st.session_state.memory.clear()
            st.session_state.chat_history = []
            st.rerun()

        # Show user context if extracted
        if st.session_state.memory.user_context:
            st.divider()
            st.caption("📋 Detected context:")
            st.caption(st.session_state.memory.user_context)

    # ── Page header ──
    st.title("⚖️ Singapore Employment Law Assistant")
    st.caption("Ask questions about your employment rights in Singapore. Powered by the Employment Act, MOM guidelines, and court judgments.")

    # ── Chat history ──
    for role, content, warnings in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)
            if warnings:
                for w in warnings:
                    st.warning(w)

    # ── Input box ──
    if user_input := st.chat_input("Ask about your employment rights..."):

        # Show user message immediately
        st.session_state.chat_history.append(("user", user_input, []))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching legal database..."):
                try:
                    reply, warnings, chunks = answer(
                        user_input,
                        st.session_state.memory,
                        verbose=False,
                    )
                    st.markdown(reply)
                    for w in warnings:
                        st.warning(w)

                    # Expandable source viewer
                    with st.expander("📚 View retrieved sources"):
                        for i, chunk in enumerate(chunks, 1):
                            meta = chunk.get("metadata", {})
                            src  = meta.get("source_type", "")
                            if src == "statute":
                                label = f"s.{meta.get('section','')} — {meta.get('section_title','')}"
                            elif src == "guideline":
                                label = meta.get("title", "")
                            elif src == "case":
                                label = meta.get("case_name", "")[:60]
                            else:
                                label = chunk.get("chunk_id", "")
                            st.caption(f"[{i}] {src.upper()}: {label}")
                            st.caption(f"Score: {chunk.get('rrf_score', 0):.4f} | URL: {meta.get('url','')}")
                            st.text(chunk.get("text", "")[:300] + "...")
                            st.divider()

                except Exception as e:
                    reply = f"An error occurred: {e}"
                    warnings = []
                    st.error(reply)

        st.session_state.chat_history.append(("assistant", reply, warnings))


# ─────────────────────────────────────────────
# CLI TEST MODE
# ─────────────────────────────────────────────

def run_cli():
    """
    CLI test mode — runs a scripted multi-turn conversation to test
    all 4 features without needing the Streamlit UI.
    """
    print("=" * 65)
    print("Chatbot CLI Test — 4-feature RAG pipeline")
    print("=" * 65)

    memory = ConversationMemory()

    # Multi-turn test: Q2 is a follow-up that needs memory to make sense
    test_turns = [
        "I earn $3,200 a month as a clerk and my employer terminated me without notice after 3 years. What are my rights?",
        "Can I also claim for unpaid overtime during those 3 years?",   # follow-up needs memory
        "What does section 38 say about hours of work?",               # direct statute lookup
    ]

    for turn_num, query in enumerate(test_turns, 1):
        print(f"\n{'='*65}")
        print(f"Turn {turn_num}: {query}")
        print("─" * 65)

        try:
            reply, warnings, chunks = answer(query, memory, verbose=True)
            print(f"\nANSWER:\n{reply}")
            if warnings:
                print("\nCITATION WARNINGS:")
                for w in warnings:
                    print(f"  {w}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 65)
    print("CLI test complete.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "streamlit" in sys.modules:
        # Running via `streamlit run chatbot.py`
        run_streamlit()
    else:
        # Running via `python chatbot.py`
        run_cli()

# When run via `streamlit run chatbot.py`, Streamlit imports the module
# at the top level, so we need this guard to trigger the UI.
# `__name__` is NOT "__main__" in that case, so we check for streamlit.
try:
    import streamlit as _st
    # If streamlit imported us (not python chatbot.py), run the UI
    if __name__ != "__main__":
        run_streamlit()
except ImportError:
    pass
