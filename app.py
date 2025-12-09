import csv
import os
from collections import defaultdict
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
from langchain_core.messages import HumanMessage
import uvicorn


from brain import app_graph

app = FastAPI()

templates = Jinja2Templates(directory="templates")


if not os.path.exists("static"):
    os.makedirs("static")
if not os.path.exists("static/css"):
    os.makedirs("static/css")

app.mount("/static", StaticFiles(directory="static"), name="static")


FLOW_TREE = defaultdict(dict)
@traceable
def load_csv_flow():
    """Parses the CSV into a dictionary for fast lookup."""
    global FLOW_TREE
    csv_path = "data/menu.csv"
    
    if not os.path.exists(csv_path):
        print("Warning: menu.csv not found.")
        return

    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_id = row['Step_ID'].strip()
            msg = row['Bot_Message'].strip()
            choice = row['User_Choice'].strip()
            next_id = row['Next_Step_ID'].strip()

            if step_id not in FLOW_TREE:
                FLOW_TREE[step_id] = {"message": msg, "options": []}
            
            # If the CSV row has choices, add them
            if choice:
                FLOW_TREE[step_id]["options"].append({
                    "label": choice,
                    "next_step": next_id
                })


load_csv_flow()



class ChatRequest(BaseModel):
    message: Optional[str] = ""
    session_id: str = "user_session_101"
    current_step_id: Optional[str] = None  
@traceable
def clean_content(content):
    """Helper to extract clean text from LangChain message."""
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



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    start_data = FLOW_TREE.get("start", {})
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "initial_message": start_data.get("message", "Welcome!"),
        "initial_options": start_data.get("options", [])
    })

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    user_msg = chat_request.message.strip()
    step_id = chat_request.current_step_id
    thread_id = chat_request.session_id

    
    if step_id and step_id in FLOW_TREE:
        node_data = FLOW_TREE[step_id]
        
       
        if step_id == "agent_handover":
             return JSONResponse(content={
                "response": node_data['message'],
                "options": [], 
                "next_step": None 
            })

        return JSONResponse(content={
            "response": node_data['message'],
            "options": node_data['options'],
            "next_step": None 
        })


    if not user_msg:
         return JSONResponse(content={"response": "I didn't catch that. Could you type it again?"})

    try:
        input_msg = HumanMessage(content=user_msg)
        config = {"configurable": {"thread_id": thread_id}}
        
        result = app_graph.invoke({"messages": [input_msg]}, config=config)

        if result and "messages" in result and len(result["messages"]) > 0:
            last_msg = result['messages'][-1]
            final_response = clean_content(last_msg.content)
        else:
            final_response = "I encountered an error connecting to the bank systems."

        # Return standard response (no flow options)
        return JSONResponse(content={
            "response": final_response,
            "options": [],
            "next_step": None
        })

    except Exception as e:
        print(f"❌ Server Error: {e}")
        return JSONResponse(content={"response": "System Error: The banking assistant is currently unavailable."})


if __name__ == "__main__":
    print("🚀 Server running on http://127.0.0.1:5009")
    uvicorn.run("app:app", host="0.0.0.0", port=5009, reload=True)