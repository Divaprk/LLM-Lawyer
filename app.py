import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import chatbot
from chatbot import ConversationMemory, answer

app = FastAPI()

# In-memory session store
sessions = {}

# Pydantic models for request bodies
class ChatRequest(BaseModel):
    message: str
    session_id: str
    role: str = "General Public"

class ClearRequest(BaseModel):
    session_id: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id
    if session_id not in sessions:
        sessions[session_id] = ConversationMemory()
    
    memory = sessions[session_id]
    
    try:
        reply, warnings, chunks, confidence = answer(
            query=req.message,
            memory=memory,
            user_role=req.role,
            verbose=False
        )
        
        sources = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            src = meta.get("source_type", "")
            
            label = ""
            if src == "statute":
                act = meta.get("act_name", "Statute")
                label = f"{act} s.{meta.get('section','')} — {meta.get('section_title','')}"
            elif src == "guideline":
                label = meta.get("title", "")
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
                
            sources.append({
                "index": i,
                "type": src.upper(),
                "label": label,
                "score": chunk.get("rrf_score", 0),
                "url": meta.get("url", ""),
                "text": chunk.get("text", "")[:300] + "..."
            })
            
        formatted_reply = chatbot.format_citations_for_display(reply)

        return {
            "reply": formatted_reply,
            "warnings": warnings,
            "confidence": confidence,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear")
async def clear_endpoint(req: ClearRequest):
    session_id = req.session_id
    if session_id in sessions:
        sessions[session_id].clear()
    return {"status": "success"}

@app.get("/api/context")
async def context_endpoint(session_id: str):
    """Return extracted user context facts for a session (for sidebar display)."""
    if session_id not in sessions:
        return {"context": ""}
    return {"context": sessions[session_id].user_context}

# Serve assets (GIFs etc.) at /assets/...
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Serve the static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Make sure to exclude the data directory, otherwise ChromaDB writing its sqlite 
    # temp files will trigger uvicorn to restart, wiping conversation memory!
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, reload_excludes=["data", "data/*"])
