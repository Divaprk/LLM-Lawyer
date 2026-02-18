import os
import json
import re
from docx import Document

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw_data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

def clean_text(text):
    if not text: return ""
    # Remove hidden characters and excessive whitespace
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def is_section_header(cell0_text):
    """
    Detects if the first cell of a row looks like a Section ID.
    Examples that match: "38", "38A", "102", "1"
    Examples that fail: "Employment Act", "Part IV", "Theft"
    """
    # Regex: Start with digit, optional letters, optional dot, end of string.
    # e.g., "38", "38A", "38."
    return re.match(r'^\d+[A-Z]?\.?$', cell0_text.strip()) is not None

def parse_docx(file_path, act_name):
    try:
        doc = Document(file_path)
    except Exception as e:
        print(f"[ERROR] Could not open {file_path}: {e}")
        return []

    chunks = []
    
    current_section_id = "Intro"
    current_text_buffer = []
    
    # We iterate through ALL tables (since the debug showed 59 of them)
    for table_idx, table in enumerate(doc.tables):
        for row in table.rows:
            cells = row.cells
            if not cells: continue
            
            # Get text from the first cell (Potential ID)
            # and the whole row (Content)
            first_cell_text = clean_text(cells[0].text)
            
            # Combine all cells in the row for the full text
            # (Use a set to avoid duplicates if cells are merged)
            unique_cells = []
            seen_text = set()
            for cell in cells:
                ct = clean_text(cell.text)
                if ct and ct not in seen_text:
                    unique_cells.append(ct)
                    seen_text.add(ct)
            full_row_text = " ".join(unique_cells)
            
            if not full_row_text: continue

            # LOGIC: If the first cell is just a number (like "38"), it's a Section Start.
            if is_section_header(first_cell_text):
                # 1. Save previous
                if current_text_buffer:
                    full_text = " ".join(current_text_buffer)
                    chunks.append({
                        "act": act_name,
                        "section": current_section_id,
                        "text": full_text
                    })
                    current_text_buffer = []
                
                # 2. Start new
                current_section_id = first_cell_text.strip('. ') # Remove trailing dots
                
                # Add the rest of the row to the buffer (Title + Content)
                # If the row was ["38", "Annual Leave"], we want "Annual Leave" in the text
                remaining_text = " ".join(unique_cells[1:]) if len(unique_cells) > 1 else ""
                if remaining_text:
                    current_text_buffer.append(remaining_text)
            
            else:
                # Just normal text, append to current section
                current_text_buffer.append(full_row_text)

    # Save the final section
    if current_text_buffer:
        full_text = " ".join(current_text_buffer)
        chunks.append({
            "act": act_name,
            "section": current_section_id,
            "text": full_text
        })
        
    return chunks

def process_all_docs():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    if not os.path.exists(RAW_DIR):
        print(f"[ERROR] Directory not found: {RAW_DIR}")
        return

    doc_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.docx')]
    total_chunks = 0
    
    print(f"--- Processing {len(doc_files)} files ---")

    for filename in doc_files:
        act_name = filename.replace('.docx', '').replace('_', ' ').title()
        file_path = os.path.join(RAW_DIR, filename)
        
        print(f"[INFO] Reading {act_name}...")
        chunks = parse_docx(file_path, act_name)
        
        if chunks:
            # Save to JSON
            output_filename = filename.replace('.docx', '.json')
            output_path = os.path.join(PROCESSED_DIR, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2)
                
            print(f"   -> Success! Extracted {len(chunks)} sections.")
            total_chunks += len(chunks)
        else:
            print(f"   -> [WARNING] No sections found in {filename}.")

    print(f"\n[DONE] Processed {total_chunks} total sections.")

if __name__ == "__main__":
    process_all_docs()