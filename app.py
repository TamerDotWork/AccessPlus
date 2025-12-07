from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from brain import app_graph
import uvicorn

# --- FastAPI app ---
app = FastAPI()

# --- Serve templates ---
templates = Jinja2Templates(directory="templates")

# --- Serve static files if any ---
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Helper: clean response content ---
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

# --- Pydantic model for incoming JSON ---
class ChatRequest(BaseModel):
    message: str
    session_id: str = "user_session_101"

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    user_message = chat_request.message.strip()
    thread_id = chat_request.session_id

    if not user_message:
        return JSONResponse(content={"response": "Please type a valid message."})

    try:
        input_msg = HumanMessage(content=user_message)
        config = {"configurable": {"thread_id": thread_id}}
        result = app_graph.invoke({"messages": [input_msg]}, config=config)

        raw_content = result['messages'][-1].content
        final_response = clean_content(raw_content)

        return JSONResponse(content={"response": final_response})

    except Exception as e:
        print(f"Server Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --- Run app ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
