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
from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY from .env

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

def get_role_instructions(role: str) -> str:
    roles = {
        "General Public": (
    "ROLE: General Public.\n"
    "Explain the law in simple everyday English.\n"
    "Avoid legal jargon.\n"
    "Focus on practical advice and what the worker should do next."
),

        "Student": (
            "ROLE: Law/HR Student.\n"
            "Explain the legal doctrine and definitions.\n"
            "Mention legal tests, statutory interpretation, and policy rationale.\n"
            "Use academic terminology where appropriate."
            "Focus on helping the reader understand the concept."
        ),

        "HR Staff": (
            "ROLE: HR Compliance Officer.\n"
            "Focus on employer obligations and compliance risk.\n"
            "Highlight required procedures, documentation, and calculations.\n"
            "Explain what the company must do to avoid violating the law."
            "Use a professional and operational tone."
        ),

        "Lawyer": (
    "ROLE: Employment Lawyer.\n"
    "Provide a technical legal analysis.\n"
    "Structure the answer as:\n"
    "1. Legal rule\n"
    "2. Legal consequence\n"
    "3. Exceptions or defenses\n"
    "Use legal terminology and statutory references."
)
    }

    return roles.get(role, roles["General Public"])


SYSTEM_PROMPT = """You are a Singapore Employment Law assistant. You help members of the public understand their rights and obligations under Singapore employment law.

RULES YOU MUST FOLLOW:
1. Base your answer ONLY on the retrieved context provided. Do not use outside knowledge.
2. Always cite your source. Use this exact format:
   - For Employment Act sections: [Employment Act s.{number}]
   - For CPF Act sections: [CPF Act s.{number}]
   - For Work Injury Compensation Act sections: [WICA s.{number}]
   - For Child Development Co-Savings Act sections: [CDCA s.{number}]
   - For Retirement and Re-employment Act sections: [RRA s.{number}]
   - For Workplace Safety and Health Act sections: [WSHA s.{number}]
   - For Personal Data Protection Act sections: [PDPA s.{number}]
   - For Employment of Foreign Manpower Act sections: [EFMA s.{number}]
   - For Workplace Fairness Act sections: [WFA s.{number}]
   - For Industrial Relations Act sections: [IRA s.{number}]
   - For TAFEP Tripartite Standards: [TS: {standard title}]
   - For Tripartite Guidelines on FWA Requests: [TG-FWAR: {section title}]
   - For WorkRight Guide sections: [WorkRight: {section title}]
   - For SLA articles: [SingaporeLegalAdvice: {article title}]
   - For court cases: [Case: {case name}]
3. IMPORTANT — Workplace Fairness Act (WFA) grace period: The Workplace Fairness Act 2025 was passed by Parliament but is in a grace period. It is NOT fully enforceable until 2027. If you cite any section of the WFA, you MUST add this disclaimer immediately after: "(Note: The Workplace Fairness Act 2025 is currently in a grace period and will only be fully enforceable from 2027.)"
4. If the user is greeting you or making casual conversation (e.g. "hello", "hi", "thanks"), respond naturally and briefly, then invite them to ask about Singapore employment law. Do NOT reference, summarise, or cite any retrieved context in this case — ignore it entirely. If the user sends nonsense, random characters, or meaningless input (e.g. "haha", "lol", "asdfgh", "???"), respond with a short, friendly message acknowledging you did not understand, and prompt them to ask a question about Singapore employment law. Do NOT reference or cite any retrieved context in this case. If the user asks something completely outside the scope of Singapore employment law (e.g. criminal law, immigration, tax, general advice unrelated to employment), do not just say you cannot help — briefly clarify that you specialise in Singapore employment law, give 2–3 examples of topics you can assist with (e.g. dismissal, leave entitlements, salary disputes, CPF, workplace safety), and invite them to ask an employment-related question. Do NOT reference or cite any retrieved context in this case either. If the user asks a legal question within scope but the retrieved context does not contain enough information to answer it, say exactly: "I'm sorry, I don't have enough information in my knowledge base to answer this question. Please consult a lawyer or visit mom.gov.sg for official guidance."
5. Never give a definitive legal ruling. Use language like "under the Employment Act...", "according to...", "you may be entitled to...". You are providing general legal information, not legal advice.
6. ROLE-BASED RESPONSE ADAPTATION:
The SAME question must be answered differently depending on the user's role.

- General Public → simple explanation and practical advice
- Student → conceptual explanation and legal definitions
- HR Staff → compliance obligations, HR procedures and Focus on compliance formulas (e.g., OT pay calculations) and risk.
- Lawyer → technical legal analysis with statutory references
FORMAT:
- Use plain, simple English that a non-lawyer can understand.
- Keep answers concise — 2 to 4 short paragraphs.
- If multiple sections or sources apply, address each briefly.
- Do not include section numbers in your prose — keep citations in brackets at the end of the relevant sentence."""


# ─────────────────────────────────────────────
# LLM CALLER
# ─────────────────────────────────────────────

def _call_llm(messages: list[dict], max_tokens: int = LLM_MAX_TOKENS) -> str:
    from openai import OpenAI
    client = OpenAI()  # uses OPENAI_API_KEY from environment
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


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
    ea_citations     = re.findall(r"\[Employment Act s\.(\w+)\]", answer)
    cpf_citations    = re.findall(r"\[CPF Act s\.(\w+)\]", answer)
    wica_citations   = re.findall(r"\[WICA s\.(\w+)\]", answer)
    cdca_citations   = re.findall(r"\[CDCA s\.(\w+)\]", answer)
    rra_citations    = re.findall(r"\[RRA s\.(\w+)\]", answer)
    wsha_citations   = re.findall(r"\[WSHA s\.(\w+)\]", answer)
    pdpa_citations   = re.findall(r"\[PDPA s\.(\w+)\]", answer)
    efma_citations   = re.findall(r"\[EFMA s\.(\w+)\]", answer)
    wfa_citations    = re.findall(r"\[WFA s\.(\w+)\]", answer)
    ira_citations    = re.findall(r"\[IRA s\.(\w+)\]", answer)
    ts_citations     = re.findall(r"\[TS:\s*([^\]]+)\]", answer)
    tgfwar_citations   = re.findall(r"\[TG-FWAR:\s*([^\]]+)\]", answer)
    workright_citations= re.findall(r"\[WorkRight:\s*([^\]]+)\]", answer)
    case_citations     = re.findall(r"\[Case:\s*([^\]]+)\]", answer)
    sla_citations      = re.findall(r"\[SingaporeLegalAdvice:\s*([^\]]+)\]", answer)

    # ── Build sets of what's actually in retrieved chunks ──
    # statute chunks keyed by (act_name_lower, section_lower)
    retrieved_statutes: dict[str, set] = {}   # act_name_key -> set of section strings
    retrieved_cases    = set()
    retrieved_sla_titles = set()
    retrieved_ts_titles  = set()

    ACT_KEYS = {
        "employment act":                    "ea",
        "central provident fund act":        "cpf",
        "work injury compensation act":      "wica",
        "child development co-savings act":  "cdca",
        "retirement and re-employment act":  "rra",
        "workplace safety and health act":   "wsha",
        "personal data protection act":      "pdpa",
        "employment of foreign manpower act":"efma",
        "workplace fairness act":            "wfa",
        "industrial relations act":          "ira",
    }

    for r in retrieved_chunks:
        meta = r.get("metadata", {})
        src  = meta.get("source_type", "")

        if src == "statute":
            act_name = meta.get("act_name", "").lower()
            section  = meta.get("section", "").lower()
            for act_fragment, key in ACT_KEYS.items():
                if act_fragment in act_name:
                    retrieved_statutes.setdefault(key, set()).add(section)

        elif src == "case":
            retrieved_cases.add(meta.get("case_name", "").lower())

        elif src == "guideline":
            title    = meta.get("title", "").lower()
            category = meta.get("category", "").lower()
            if "tripartite standard" in category:
                retrieved_ts_titles.add(title)
            else:
                retrieved_sla_titles.add(title)

    # ── Verify each citation ──
    def _statute_warn(label: str, sec: str, act_key: str):
        if sec.lower() not in retrieved_statutes.get(act_key, set()):
            warnings.append(
                f"⚠️ Citation [{label} s.{sec}] could not be verified "
                f"against retrieved sources."
            )

    for sec in ea_citations:
        _statute_warn("Employment Act", sec, "ea")
    for sec in cpf_citations:
        _statute_warn("CPF Act", sec, "cpf")
    for sec in wica_citations:
        _statute_warn("WICA", sec, "wica")
    for sec in cdca_citations:
        _statute_warn("CDCA", sec, "cdca")
    for sec in rra_citations:
        _statute_warn("RRA", sec, "rra")
    for sec in wsha_citations:
        _statute_warn("WSHA", sec, "wsha")
    for sec in pdpa_citations:
        _statute_warn("PDPA", sec, "pdpa")
    for sec in efma_citations:
        _statute_warn("EFMA", sec, "efma")
    for sec in wfa_citations:
        _statute_warn("WFA", sec, "wfa")
    for sec in ira_citations:
        _statute_warn("IRA", sec, "ira")

    # WFA grace period programmatic notice
    if wfa_citations:
        warnings.append(
            "ℹ️ The Workplace Fairness Act 2025 is currently in a grace period "
            "and will only be fully enforceable from 2027. Employers are encouraged "
            "to comply early, but enforcement has not yet commenced."
        )

    for title in ts_citations:
        title_lower = title.lower().strip()
        found = any(title_lower in rt or rt in title_lower for rt in retrieved_ts_titles)
        if not found:
            warnings.append(
                f"⚠️ Citation [TS: {title}] could not be verified "
                f"against retrieved sources."
            )

    # Build TG-FWAR title set from retrieved guideline chunks
    retrieved_tgfwar_titles: set = set()
    for r in retrieved_chunks:
        meta = r.get("metadata", {})
        if (meta.get("source_type") == "guideline"
                and "tripartite guideline" in meta.get("category", "").lower()):
            retrieved_tgfwar_titles.add(meta.get("title", "").lower())

    for title in tgfwar_citations:
        title_lower = title.lower().strip()
        found = any(title_lower in rt or rt in title_lower for rt in retrieved_tgfwar_titles)
        if not found:
            warnings.append(
                f"⚠️ Citation [TG-FWAR: {title}] could not be verified "
                f"against retrieved sources."
            )

    # WorkRight guide citations
    retrieved_wr_titles: set = set()
    for r in retrieved_chunks:
        meta = r.get("metadata", {})
        if (meta.get("source_type") == "guideline"
                and "workright" in meta.get("category", "").lower()):
            retrieved_wr_titles.add(meta.get("title", "").lower())

    for title in workright_citations:
        title_lower = title.lower().strip()
        found = any(title_lower in rt or rt in title_lower for rt in retrieved_wr_titles)
        if not found:
            warnings.append(
                f"⚠️ Citation [WorkRight: {title}] could not be verified "
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
# FEATURE 6: Confidence Scoring
# ─────────────────────────────────────────────

def compute_confidence_score(
    retrieved_chunks: list[dict],
    verified_answer: str,
    warnings: list[str],
) -> dict:
    """
    Compute a confidence score (0–100) for the generated answer.

    Factors considered:
      1. Retrieval quality  — average RRF score of top chunks (higher = better match)
      2. Source diversity   — mixture of statutes, cases, guidelines is more reliable
      3. Citation warnings  — each unverified citation penalises the score
      4. Answer length      — very short answers may indicate low-confidence fallbacks

    Returns a dict:
        {
            "score": int (0–100),
            "label": "High" | "Medium" | "Low",
            "color": "green" | "orange" | "red",
            "breakdown": { ... individual sub-scores for debugging }
        }
    """
    if not retrieved_chunks:
        return {
            "score": 0,
            "label": "Low",
            "color": "red",
            "breakdown": {"reason": "No chunks retrieved"},
        }

    # ── Sub-score 1: Retrieval quality (0–40 pts) ──
    # RRF scores are typically in the 0.01–0.07 range; we normalise against 0.06 as a ceiling
    scores = [c.get("rrf_score", 0) for c in retrieved_chunks]
    avg_score = sum(scores) / len(scores) if scores else 0
    RRF_CEILING = 0.06
    retrieval_pts = min(40, int((avg_score / RRF_CEILING) * 40))

    # ── Sub-score 2: Source diversity (0–25 pts) ──
    source_types = {c.get("metadata", {}).get("source_type", "unknown") for c in retrieved_chunks}
    diversity_pts = min(25, len(source_types) * 8)   # 1 type=8, 2=16, 3+=24 (capped at 25)

    # ── Sub-score 3: Citation penalty (–10 per unverified warning) ──
    unverified_warnings = [w for w in warnings if "could not be verified" in w]
    citation_penalty = min(30, len(unverified_warnings) * 10)

    # ── Sub-score 4: Answer completeness (0–15 pts) ──
    word_count = len(verified_answer.split())
    if word_count >= 80:
        completeness_pts = 15
    elif word_count >= 40:
        completeness_pts = 8
    else:
        completeness_pts = 0   # very short = likely a fallback/refusal

    raw = retrieval_pts + diversity_pts + completeness_pts - citation_penalty
    final_score = max(0, min(100, raw))

    if final_score >= 70:
        label, color = "High", "green"
    elif final_score >= 40:
        label, color = "Medium", "orange"
    else:
        label, color = "Low", "red"

    return {
        "score": final_score,
        "label": label,
        "color": color,
        "breakdown": {
            "retrieval_pts": retrieval_pts,
            "diversity_pts": diversity_pts,
            "completeness_pts": completeness_pts,
            "citation_penalty": -citation_penalty,
            "chunks_retrieved": len(retrieved_chunks),
            "avg_rrf_score": round(avg_score, 5),
            "source_types": list(source_types),
        },
    }

def calculate_employment_payments(data: dict) -> str:
    """
    Deterministic calculation for PH and OT pay.
    Formula: (12 * Monthly Basic) / (52 * 44) for Hourly Rate.
    Formula: (12 * Monthly Basic) / (52 * Days Per Week) for Daily Rate.
    """
    try:
        salary = float(str(data.get("salary", 0)).replace(",", "").replace("$", ""))
        days_per_week = float(data.get("work_days_per_week", 5))
        
        # Calculate Daily Rate (for Public Holidays)
        daily_rate = (12 * salary) / (52 * days_per_week)
        
        # Calculate Hourly Rate (for Overtime)
        # Standard work week is 44 hours in Singapore
        hourly_rate = (12 * salary) / (52 * 44)
        
        results = []
        if data.get("ph_worked"):
            ph_pay = daily_rate * float(data.get("ph_worked"))
            results.append(f"Public Holiday Pay: ${ph_pay:.2f} (for {data['ph_worked']} days)")
            
        if data.get("ot_hours"):
            # OT is 1.5x hourly rate
            ot_pay = (hourly_rate * 1.5) * float(data.get("ot_hours"))
            results.append(f"Overtime Pay (1.5x): ${ot_pay:.2f} (for {data['ot_hours']} hours)")
            
        return "\n".join(results) if results else "Could not calculate: missing values."
    except Exception as e:
        return f"Calculation error: {str(e)}"
    

CALC_EXTRACTION_PROMPT = """
Extract the following variables for a Singapore legal pay calculation from the text below.
Return ONLY a JSON object. If a value is unknown, use null.

Variables:
- salary: monthly basic salary amount
- work_days_per_week: number of days worked per week (default to 5 if not mentioned)
- ph_worked: number of public holidays worked
- ot_hours: number of overtime hours worked

Text: "{text}"
JSON:"""


def format_citations_for_display(text: str) -> str:
    """
    Convert inline citation brackets to superscript numbers and append
    a numbered reference list after the response body.

    e.g.  "...blah [SingaporeLegalAdvice: Title]..."
      →   "...blah<sup>[1]</sup>...\n\n---\n**References:**\n1. SingaporeLegalAdvice: Title"
    """
    citation_pattern = re.compile(
        r'\[(?:'
        r'Employment Act s\.\w+'
        r'|CPF Act s\.\w+'
        r'|WICA s\.\w+'
        r'|CDCA s\.\w+'
        r'|RRA s\.\w+'
        r'|WSHA s\.\w+'
        r'|PDPA s\.\w+'
        r'|EFMA s\.\w+'
        r'|WFA s\.\w+'
        r'|IRA s\.\w+'
        r'|TS:\s*[^\]]+'
        r'|TG-FWAR:\s*[^\]]+'
        r'|WorkRight:\s*[^\]]+'
        r'|SingaporeLegalAdvice:\s*[^\]]+'
        r'|Case:\s*[^\]]+'
        r')\]'
    )

    seen: dict[str, int] = {}
    counter = [0]

    def _replace(match: re.Match) -> str:
        citation = match.group(0)
        if citation not in seen:
            counter[0] += 1
            seen[citation] = counter[0]
        return f'<sup>[{seen[citation]}]</sup>'

    formatted = citation_pattern.sub(_replace, text)

    if seen:
        refs = "\n".join(
            f"{n}. {cite[1:-1]}"
            for cite, n in sorted(seen.items(), key=lambda x: x[1])
        )
        formatted += f"\n\n---\n**References:**\n{refs}"

    return formatted


# ─────────────────────────────────────────────
# FEATURE 5: Conversation Memory
# ─────────────────────────────────────────────

INTAKE_CLASSIFIER_PROMPT = """You are a Singapore employment law intake assistant.

A user has asked: "{query}"

Their known context so far: "{context}"

Your job:
1. Classify the query topic into ONE of these categories:
   - termination (fired, dismissed, notice period, wrongful dismissal)
   - salary_dispute (unpaid salary, deductions, overtime pay)
   - leave (annual leave, sick leave, maternity/paternity leave)
   - workplace_safety (injury, accident, WSHA, unsafe conditions)
   - discrimination (unfair treatment, WFA, TAFEP, protected characteristics)
   - contract (employment contract terms, probation, restraint of trade)
   - cpf (CPF contributions, withdrawal, shortfall)
   - flexible_work (FWA request, work from home, flexible hours)
   - reemployment (retirement age, re-employment obligation, RRA)
   - foreign_worker (work pass, EP, S Pass, EFMA)
   - general (unclear or multiple topics)

2. Decide if clarifying questions are ACTUALLY NEEDED by asking: would the answer meaningfully change based on the user's personal details?

Set "intake_needed": false if:
   - The answer is the same regardless of job type, salary, or duration (e.g. general working hours limits, notice period rules, what a law says)
   - The question is asking what the law says in general ("is X legal?", "what does section X say?", "what are my rights regarding X?")
   - The user is asking for a general explanation or overview
   - Enough context is already known

Set "intake_needed": true ONLY if:
   - The answer changes significantly based on job type (workman vs executive)
   - The entitlement amount depends on employment duration (e.g. severance, leave days)
   - EA coverage depends on salary amount (e.g. salary dispute, overtime claims)
   - The user's specific situation (fired, injured, underpaid) requires personalised advice

3. If intake_needed is true, return 2-3 clarifying questions. Otherwise return an empty list.
   - Skip any question whose answer is already known from the context.
   - Questions must directly affect which law or entitlement applies.

Return ONLY a JSON object like this (no other text):
{{
  "topic": "<category>",
  "intake_needed": true,
  "questions": ["Question 1?", "Question 2?"]
}}"""


def situation_intake(query: str, memory) -> list[str]:
    """
    Dynamically generate clarifying questions based on the user's query topic.
    Uses LLM to classify intent and select relevant questions.
    Skips questions already answered in memory.user_context.
    Returns [] if no clarification needed (e.g. casual chat, already has context).
    """
    # Don't ask intake questions for greetings or very short inputs
    if len(query.strip().split()) <= 2:
        return []

    # Don't ask intake questions if we already have sufficient context
    known_facts = memory.user_context
    known_count = known_facts.count(";") + (1 if known_facts else 0)
    if known_count >= 2:
        return []

    prompt = INTAKE_CLASSIFIER_PROMPT.format(
        query=query,
        context=known_facts if known_facts else "None"
    )

    try:
        raw = _call_llm([{"role": "user", "content": prompt}], max_tokens=300)
        # Strip markdown fences if present
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)

        # Skip intake entirely if the LLM says it's not needed
        if not parsed.get("intake_needed", True):
            return []

        questions = parsed.get("questions", [])

        # Filter out questions whose answers are already in context
        context_lower = known_facts.lower()
        filtered = []
        for q in questions:
            q_lower = q.lower()
            # Skip if answer keywords already exist in context
            if "job type" in q_lower and "job type:" in context_lower:
                continue
            if ("long" in q_lower or "duration" in q_lower or "how long" in q_lower) and "employment duration:" in context_lower:
                continue
            if "contract" in q_lower and "written contract:" in context_lower:
                continue
            if "salary" in q_lower and "salary:" in context_lower:
                continue
            filtered.append(q)

        return filtered[:3]  # cap at 3 questions max

    except Exception as e:
        print(f"  [WARN] Situation intake LLM call failed: {e}")
        return []


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
        self.pending_query = ""        # original question saved while waiting for intake answers

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

        # Written contract (was mistakenly outside the save block before)
        if re.search(r"\bwritten contract\b|\bsigned contract\b|\bemployment contract\b|\boffer letter\b|\bhave a contract\b", user_message, re.IGNORECASE):
            facts.append("written contract: yes")
        elif re.search(r"\bno contract\b|\bdidn't sign\b|\bdid not sign\b|\bverbal agreement\b", user_message, re.IGNORECASE):
            facts.append("written contract: no")

        if facts:
            # Append new facts, avoid duplicates
            new_context = "; ".join(facts)
            if new_context not in self.user_context:
                self.user_context = (self.user_context + "; " + new_context).strip("; ")

    def clear(self):
        self.history      = []
        self.user_context = ""
        self.pending_query = ""
        


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
            act_name = meta.get("act_name", "Employment Act")
            label = (f"[SOURCE {i}] {act_name} s.{meta.get('section','')} "
                     f"— {meta.get('section_title','')}")
        elif src == "guideline":
            category = meta.get("category", "")
            title    = meta.get("title", "")
            if "Tripartite Standard" in category:
                label = f"[SOURCE {i}] TS: {title}"
            elif "Tripartite Guideline" in category:
                label = f"[SOURCE {i}] TG-FWAR: {title}"
            elif "WorkRight" in category:
                label = f"[SOURCE {i}] WorkRight: {title}"
            else:
                label = f"[SOURCE {i}] SingaporeLegalAdvice: {title}"
        elif src == "case":
            label = (f"[SOURCE {i}] Case: {meta.get('case_name','')} "
                     f"({meta.get('court','')}, {meta.get('year','')})")
        else:
            label = f"[SOURCE {i}]"

        blocks.append(f"{label}\n{text}")

    return "\n\n---\n\n".join(blocks)


CONTEXT_EXTRACT_PROMPT = """You are extracting key employment facts from a user's message.

User said: "{message}"

Extract any of these facts if mentioned (return null if not mentioned):
- job_type: their role/type (e.g. "clerk", "manager", "workman", "executive", "engineer")
- employment_duration: how long they worked (e.g. "3 years", "6 months")
- written_contract: do they have a written contract? (yes/no)
- salary: their salary amount (e.g. "$2,500/month")
- still_employed: are they currently still working there? (yes/no)

Return ONLY a JSON object, no other text:
{{"job_type": null, "employment_duration": null, "written_contract": null, "salary": null, "still_employed": null}}"""


def _llm_extract_context(query: str, memory) -> None:
    """
    Use LLM to extract facts from free-text answers (e.g. responses to clarifying questions).
    Updates memory.user_context with any new facts found.
    Only runs if the query looks like an answer to a clarifying question (short, factual).
    """
    # Only run on shorter replies that look like answers, not full questions
    word_count = len(query.strip().split())
    if word_count > 40:
        return

    try:
        raw = _call_llm(
            [{"role": "user", "content": CONTEXT_EXTRACT_PROMPT.format(message=query)}],
            max_tokens=150
        )
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        extracted = json.loads(raw)

        new_facts = []
        ctx_lower = memory.user_context.lower()

        if extracted.get("job_type") and "job type:" not in ctx_lower:
            new_facts.append(f"job type: {extracted['job_type']}")
        if extracted.get("employment_duration") and "employment duration:" not in ctx_lower:
            new_facts.append(f"employment duration: {extracted['employment_duration']}")
        if extracted.get("written_contract") and "written contract:" not in ctx_lower:
            new_facts.append(f"written contract: {extracted['written_contract']}")
        if extracted.get("salary") and "salary:" not in ctx_lower:
            new_facts.append(f"salary: {extracted['salary']}")
        if extracted.get("still_employed") and "still employed:" not in ctx_lower:
            new_facts.append(f"still employed: {extracted['still_employed']}")

        if new_facts:
            addition = "; ".join(new_facts)
            memory.user_context = (memory.user_context + "; " + addition).strip("; ")

    except Exception as e:
        print(f"  [WARN] LLM context extraction failed: {e}")


def answer(
    query: str,
    memory: ConversationMemory,
    user_role: str = "General Public",
    verbose: bool = False,
) -> tuple[str, list[str], list[dict]]:
    """
    Full RAG pipeline for one user turn.

    Returns:
        (answer_text, citation_warnings, retrieved_chunks)
    """
    from retrieval import retrieve

    # ── 1. Extract user context facts (regex for structured data) ──
    memory.extract_user_context(query)

    # ── 1b. LLM-based extraction for free-text answers ──
    _llm_extract_context(query, memory)

    # ── 2. Situation Intake (ask clarification questions before retrieval) ──
    missing_questions = situation_intake(query, memory)

    if missing_questions:
        # Save the original question so retrieval uses it once the user answers
        if not memory.pending_query:
            memory.pending_query = query
        clarification_reply = (
            "I need a few details before I can answer more accurately:\n\n- "
            + "\n- ".join(missing_questions)
        )
        return clarification_reply, [], [], {"score": 0, "label": "Low", "color": "red", "breakdown": {"reason": "Awaiting clarification"}}

    # If user just answered our clarifying questions, retrieve against the original question
    effective_question = memory.pending_query if memory.pending_query else query
    memory.pending_query = ""  # clear it now that we're proceeding
    if any(word in query.lower() for word in ["calculate", "how much", "pay", "salary", "math"]):
        raw_calc_data = _call_llm([{"role": "user", "content": CALC_EXTRACTION_PROMPT.format(text=query + " " + memory.user_context)}])
        try:
            calc_json = json.loads(raw_calc_data.strip().removeprefix("```json").removesuffix("```"))
            # Only perform math if we have enough data (salary + (PH or OT))
            if calc_json.get("salary") and (calc_json.get("ph_worked") or calc_json.get("ot_hours")):
                math_result = calculate_employment_payments(calc_json)
                # Add this math result to the prompt context so the LLM can explain it
                query += f"\n[SYSTEM CALCULATION RESULT: {math_result}]"
        except:
            pass # Fallback to standard RAG if extraction fails
    # ── 3. LLM query rewriting ──
    conv_context = memory.get_context_string()
    effective_query = rewrite_query_llm(effective_question, conv_context)
    if verbose:
        print(f"  [REWRITE] {effective_question!r}\n        → {effective_query!r}")

    # Append known user context to improve retrieval
    if memory.user_context:
        effective_query += f" {memory.user_context}"

    # ── 4. Hybrid retrieval ──
    raw_results = retrieve(effective_query, top_k=TOP_K_RETRIEVE, rewrite=False)

    # ── 5. Parent-document escalation ──
    results = escalate_to_parent(raw_results)

    if verbose:
        print(f"  [RETRIEVE] Top results:")
        for r in results:
            m = r.get("metadata", {})
            print(f"    {m.get('source_type','')} | s.{m.get('section','')} {m.get('title',m.get('case_name',''))[:40]} | score={r.get('rrf_score',0):.4f}")

    # ── 6. Build prompt ──
    context_block = build_context_block(results)
    user_content  = f"""
    USER ROLE: {user_role}

    USER BACKGROUND: {memory.user_context if memory.user_context else "None provided."}
    
    User question: {effective_question}

Retrieved context:
{context_block}

INSTRUCTION: 
- If User Background is 'None', identify the missing 'Binary Gate' facts (Job Type/Salary) and ask for them briefly.
- If User Background is present, use it to filter the legal context and provide the FINAL ANSWER to the original query now.
- Do not ask more questions once the 'Binary Gate' is cleared.
Answer the question based ONLY on the context above. Cite each source used."""

    # Add user context if available
    if memory.user_context:
        user_content = f"User background: {memory.user_context}\n\n" + user_content##<<-- here change
    role_context = get_role_instructions(user_role)
    # Build full message list: system + history + current question
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT+ "\n\n" + role_context}]
        + memory.get_messages()
        + [{"role": "user", "content": user_content}]
    )

    # ── 6. LLM generation ──
    llm_answer = _call_llm(messages)

    # ── 7. Citation verification ──
    verified_answer, warnings = verify_citations(llm_answer, results)

    # ── 8. Confidence scoring ──
    confidence = compute_confidence_score(results, verified_answer, warnings)
    if verbose:
        print(f"  [CONFIDENCE] Score={confidence['score']} ({confidence['label']}) | breakdown={confidence['breakdown']}")

    # ── 9. Update conversation memory ──
    memory.add_turn("user", query)          # store original query (not rewritten)
    memory.add_turn("assistant", llm_answer)

    return verified_answer, warnings, results, confidence


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

    st.markdown(
        """
        <style>
        .disclaimer-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #2a2a2a;
            color: #cccccc;
            text-align: center;
            padding: 10px 16px;
            font-size: 0.85rem;
            z-index: 9999;
            border-top: 1px solid #555555;
        }
        </style>
        <div class="disclaimer-footer">
            ⚠️ This is general information only and does not constitute legal advice.
            For your specific situation, please consult a qualified employment lawyer or the
            <a href="https://www.mom.gov.sg" target="_blank" style="color:#cccccc; text-decoration: underline;">Ministry of Manpower (mom.gov.sg)</a>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Session state ──
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of (role, content, warnings)

    # ── Sidebar ──
    with st.sidebar:
        st.title("⚖️ SG Employment Law")
        user_role = st.selectbox(
            "I am a...",
            options=["General Public", "Student", "HR Staff", "Lawyer"],
            index=0
        )
        st.divider()
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
            st.markdown(content, unsafe_allow_html=True)
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
                    reply, warnings, chunks, confidence = answer(
                        user_input,
                        st.session_state.memory,
                        user_role=user_role,
                        verbose=False,
                    )
                    reply = format_citations_for_display(reply)
                    st.markdown(reply, unsafe_allow_html=True)

                    # ── Confidence badge ──
                    badge_color = {"green": "#2e7d32", "orange": "#e65100", "red": "#b71c1c"}[confidence["color"]]
                    st.markdown(
                        f'<span style="background:{badge_color};color:white;padding:3px 10px;'
                        f'border-radius:12px;font-size:0.8rem;font-weight:600;">'
                        f'🎯 Confidence: {confidence["label"]} ({confidence["score"]}/100)'
                        f'</span>',
                        unsafe_allow_html=True,
                    )

                    for w in warnings:
                        st.warning(w)

                    # Expandable source viewer
                    with st.expander("📚 View retrieved sources"):
                        for i, chunk in enumerate(chunks, 1):
                            meta = chunk.get("metadata", {})
                            src  = meta.get("source_type", "")
                            if src == "statute":
                                act = meta.get("act_name", "Statute")
                                label = f"{act} s.{meta.get('section','')} — {meta.get('section_title','')}"
                            elif src == "guideline":
                                label    = meta.get("title", "")
                                category = meta.get("category", "")
                                if category == "Tripartite Standard":
                                    label = f"[TS] {label}"
                                elif category == "Tripartite Guideline":
                                    label = f"[TG-FWAR] {label}"
                                elif category == "WorkRight Guide":
                                    label = f"[WorkRight] {label}"
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
            reply, warnings, chunks, confidence = answer(query, memory, verbose=True)
            print(f"\nANSWER:\n{reply}")
            print(f"\nCONFIDENCE: {confidence['label']} ({confidence['score']}/100)")
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
    run_cli()