"""
fetch_employment_act.py — Parse Employment Act from MHTML file
==============================================================

USAGE
-----
    python fetch_employment_act.py
    python fetch_employment_act.py "path/to/your_file.mhtml"

Output: data/chunks/employment_act.jsonl  (overwrites previous)

HOW TO GET THE MHTML FILE
--------------------------
1. Open https://sso.agc.gov.sg/Act/EmA1968 in Chrome
2. Wait for the full Act to load (scroll through it once to be safe)
3. Press Ctrl+S → choose "Webpage, Single File (*.mhtml)"
4. Save it as:  data/raw/employment_act.mhtml
5. Run this script

WHY MHTML AND NOT HTML?
-----------------------
SSO is a JavaScript SPA. A plain "HTML only" save only captures the
Table of Contents skeleton. MHTML saves the fully-rendered DOM with
all JS-injected section body text included.

HTML STRUCTURE (verified by inspection of real MHTML file)
-----------------------------------------------------------
  <td class="part" id="P1{N}-">           one per Part (18 total)
    <div class="partNo">PART 2</div>       part number
    <table>CONTRACTS OF SERVICE</table>    part name (next sibling of partNo)
    <div class="prov1">                    one per Section (154 total)
      <td class="prov1Hdr" id="pr14-">    section number + title
        <span class="">Dismissal</span>
      </td>
      ... subsection rows with full legal text ...
    </div>
  </td>
"""

import sys
import re
import json
import email
import hashlib
from pathlib import Path
from datetime import datetime

from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEFAULT_MHTML = "data/raw/employment_act.mhtml"
OUTPUT_PATH   = Path("data/chunks/employment_act.jsonl")
SSO_URL       = "https://sso.agc.gov.sg/Act/EmA1968"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1: Extract HTML from MHTML container
# ─────────────────────────────────────────────

def extract_html_from_mhtml(path: Path) -> str:
    """
    MHTML is a multipart MIME format (like an email with attachments).
    The main page HTML is always the first text/html part.
    Python's built-in email module handles it cleanly.
    """
    print(f"[MHTML] Reading: {path}")
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f)

    for part in msg.walk():
        if part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True)
            charset = part.get_content_charset() or "utf-8"
            html    = payload.decode(charset, errors="replace")
            print(f"  HTML part extracted: {len(html):,} chars")
            return html

    raise ValueError("No text/html part found in MHTML. Is the file valid?")


# ─────────────────────────────────────────────
# STEP 2: Parse sections into chunks
# ─────────────────────────────────────────────

def parse_employment_act(html: str) -> list[dict]:
    """
    Parse Employment Act HTML into one chunk per section.

    Outer loop: <td class="part"> — one per Part (gives part name)
    Inner loop: <div class="prov1"> inside each part — one per Section
    """
    print("\n[PARSE] Parsing sections...")
    soup = BeautifulSoup(html, "lxml")

    chunks = []

    for ptd in soup.find_all("td", class_="part"):

        # ── Get Part name ──────────────────────────────────────────
        # <div class="partNo"> holds "PART 2"
        # Its next sibling <table> holds "CONTRACTS OF SERVICE"
        partno_div = ptd.find("div", class_="partNo")
        if not partno_div:
            continue

        partno_text   = partno_div.get_text(strip=True)        # "PART 2"
        name_table    = partno_div.find_next_sibling("table")
        partname_text = name_table.get_text(strip=True) if name_table else ""
        part_label    = f"{partno_text} {partname_text}".strip()

        # ── Parse each section inside this Part ───────────────────
        for prov1_div in ptd.find_all("div", class_="prov1"):
            hdr = prov1_div.find("td", class_="prov1Hdr")
            if not hdr:
                continue

            # Section ID: "pr14-" → number "14", "pr18A-" → "18A"
            sec_id  = hdr.get("id", "")
            sec_num = re.sub(r"^pr(.+)-$", r"\1", sec_id)

            # Section title: last <span> in the header td
            # (first span is the section number, last is the title)
            spans     = hdr.find_all("span")
            sec_title = spans[-1].get_text(strip=True) if spans else hdr.get_text(strip=True)

            # Full text of the section — all subsections, never split
            full_text = prov1_div.get_text(separator="\n", strip=True)

            if len(full_text.strip()) < 20:
                continue

            is_repealed = bool(re.search(r"repealed|deleted", full_text, re.IGNORECASE))
            cross_refs  = sorted(set(
                re.findall(r"\bsections?\s+(\d+\w*)", full_text, re.IGNORECASE)
            ))

            # embed_text prefixes metadata so the embedding model can
            # match "hours of work rules" → Section 38 even when the body
            # text doesn't use those exact words
            embed_text = (
                f"Employment Act 1968, {part_label}, "
                f"Section {sec_num} — {sec_title}:\n\n{full_text}"
            )

            chunks.append({
                "chunk_id":      f"EA_{hashlib.md5(embed_text.encode()).hexdigest()[:8]}",
                "source_type":   "statute",
                "act_name":      "Employment Act 1968",
                "part":          part_label,
                "section":       sec_num,
                "section_title": sec_title,
                "is_repealed":   is_repealed,
                "text":          full_text,
                "embed_text":    embed_text,
                "cross_refs":    cross_refs,
                "url":           f"{SSO_URL}#{sec_id}",
                "scraped_at":    datetime.utcnow().isoformat(),
            })

    return chunks


# ─────────────────────────────────────────────
# STEP 3: Quality check
# ─────────────────────────────────────────────

def quality_check(chunks: list[dict]):
    """Spot-check key sections and report stats."""
    total    = len(chunks)
    live     = [c for c in chunks if not c["is_repealed"]]
    repealed = [c for c in chunks if c["is_repealed"]]
    stubs    = [c for c in chunks if len(c["text"]) < len(c["section_title"]) + 30]
    short    = [c for c in chunks if len(c["text"]) < 100 and not c["is_repealed"]]

    print("\n[QC] Quality Report")
    print(f"  Total chunks:       {total}  (expect ~154)")
    print(f"  Live sections:      {len(live)}  (expect ~146)")
    print(f"  Repealed/deleted:   {len(repealed)}")
    print(f"  Stub chunks:        {len(stubs)}  ← must be 0")
    print(f"  Short live (<100c): {len(short)}  ← should be near 0")

    spot_checks = {
        "2":   "Interpretation",
        "14":  "Dismissal",
        "38":  "Hours of work",
        "76":  "Length of benefit period",
        "88A": "Annual leave",
    }
    print("\n  Spot checks:")
    print(f"  {'Sec':<6} {'Part':<38} {'Title':<30} {'Chars':>6}")
    print(f"  {'-'*6} {'-'*38} {'-'*30} {'-'*6}")
    sec_map = {c["section"]: c for c in chunks}
    for sec, expected in spot_checks.items():
        c = sec_map.get(sec)
        if c:
            ok = "✓" if expected.lower() in c["section_title"].lower() else "?"
            print(f"  {sec:<6} {c['part'][:38]:<38} {c['section_title'][:30]:<30} {len(c['text']):>5}c {ok}")
        else:
            print(f"  {sec:<6} NOT FOUND ✗")

    if stubs:
        print(f"\n  ⚠️  Stub sections (no body text):")
        for c in stubs[:5]:
            print(f"    s.{c['section']}: {repr(c['text'][:80])}")

    overall = "✅ GOOD" if total >= 100 and len(stubs) == 0 else "⚠️  CHECK ISSUES ABOVE"
    print(f"\n  Overall: {overall}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # Accept optional path as command-line arg
    mhtml_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_MHTML)

    print("=" * 60)
    print("Employment Act Parser — MHTML Edition")
    print("=" * 60)

    # Auto-discover if default path doesn't exist
    if not mhtml_path.exists():
        candidates = (
            list(Path("data/raw").glob("*.mhtml"))
            + list(Path("data/raw").glob("*.mht"))
            + list(Path(".").glob("*.mhtml"))
            + list(Path(".").glob("*.mht"))
        )
        if candidates:
            mhtml_path = candidates[0]
            print(f"[AUTO] Found: {mhtml_path}")
        else:
            print(f"[ERROR] No MHTML file found at: {mhtml_path}")
            print()
            print("Steps to get the file:")
            print("  1. Open https://sso.agc.gov.sg/Act/EmA1968 in Chrome")
            print("  2. Wait for the full Act to load")
            print("  3. Ctrl+S → 'Webpage, Single File (*.mhtml)'")
            print(f"  4. Save to: {Path(DEFAULT_MHTML).resolve()}")
            print("  5. Re-run: python fetch_employment_act.py")
            return

    # Parse
    html   = extract_html_from_mhtml(mhtml_path)
    chunks = parse_employment_act(html)

    if not chunks:
        print("\n[ERROR] No chunks produced.")
        print("The MHTML structure may differ from expected. Check the file.")
        return

    # Quality check
    quality_check(chunks)

    # Save
    print(f"\n[SAVE] {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"  Saved {len(chunks)} chunks.")
    print("=" * 60)


if __name__ == "__main__":
    main()