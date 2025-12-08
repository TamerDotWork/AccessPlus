
# ------------------ app.py ------------------
"""
FastAPI wrapper with pre/post sanitization and safe handling of responses.
- Cleans user input
- Detects prompt injection and rejects it
- Calls the graph and returns sanitized result
- Provides hook for human-in-the-loop approval for high-risk requests (placeholder)
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from brain import app_graph, sanitize_user_input, detect_prompt_injection, post_process_response
import uvicorn
import logging

logger = logging.getLogger("banking_app")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Quick rate limiter placeholder (implement Redis or token-bucket in prod)
from collections import defaultdict
RATE_LIMIT = defaultdict(int)
RATE_LIMIT_THRESHOLD = 30  # very naive

class ChatRequest(BaseModel):
    message: str
    session_id: str = "user_session_101"

# Helper: clean response content - accepts string or structured content
def clean_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and 'text' in part:
                text_parts.append(part['text'])
            else:
                text_parts.append(str(part))
        return " ".join(text_parts)
    return str(content)

# Simple heuristic to decide if response contains an action that needs human approval
HIGH_RISK_KEYWORDS = ["transfer", "send money", "wire", "delete account", "close account"]

def needs_human_approval(user_input: str) -> bool:
    ui = user_input.lower()
    return any(k in ui for k in HIGH_RISK_KEYWORDS)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    raw_user_message = chat_request.message or ""
    thread_id = chat_request.session_id

    # Rate limiting (naive)
    RATE_LIMIT[thread_id] += 1
    if RATE_LIMIT[thread_id] > RATE_LIMIT_THRESHOLD:
        return JSONResponse(content={"error": "Rate limit exceeded."}, status_code=429)

    # Pre-processing & sanitization
    user_message = sanitize_user_input(raw_user_message)
    if not user_message:
        return JSONResponse(content={"response": "Please type a valid message."})

    # Detect prompt injection
    if detect_prompt_injection(user_message):
        logger.warning(f"Prompt injection attempt blocked for session {thread_id}")
        return JSONResponse(content={"response": "Unsafe input detected — please rephrase."})

    # If this is a high-risk request, mark for manual approval and do NOT execute
    if needs_human_approval(user_message):
        # Placeholder: push to human queue, store audit entry in DB, return pending message
        logger.info(f"High-risk request routed to human for session {thread_id}")
        # In production: create a task for ops, notify via channel, wait for approve
        return JSONResponse(content={"response": "This action requires manual approval. We have created a request and will follow up."})

    try:
        input_msg = HumanMessage(content=user_message)
        config = {"configurable": {"thread_id": thread_id}}
        result = app_graph.invoke({"messages": [input_msg]}, config=config)

        raw_content = result['messages'][-1].content
        final_response = clean_content(raw_content)

        # extra post-processing / redaction on server-side
        final_response = post_process_response(final_response)

        # Audit log (in real system, write to DB or tamper-evident store)
        logger.info(f"Chat response for {thread_id}: {final_response}")

        return JSONResponse(content={"response": final_response})

    except Exception as e:
        logger.exception("Server Error in chat_endpoint")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    # Run with command: python app.py
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
