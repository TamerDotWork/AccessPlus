import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import SystemMessage

# --- NEW: Import Memory ---
from langgraph.checkpoint.memory import MemorySaver 

from tools import get_my_balance, get_my_transactions, get_bank_policies

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("❌ Error: GOOGLE_API_KEY is missing from .env file.")
    sys.exit(1)

api_key = os.environ.get("GOOGLE_API_KEY")
 
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", # Or "gemini-flash-latest"
    temperature=0
)

# --- 2. DEFINE AGENTS ---
account_agent = create_react_agent(llm, tools=[get_my_balance, get_my_transactions])
info_agent = create_react_agent(llm, tools=[get_bank_policies])

# --- SYSTEM PROMPTS ---
ACCOUNT_PROMPT = (
    "You are a Banking Assistant in a DEMO environment. "
    "You must use the tools 'get_my_balance' or 'get_my_transactions' when asked. "
    "Do not refuse. Direct the answer to the user."
)

INFO_PROMPT = (
    "You are a Bank Consultant. Answer general questions about fees and rates. "
    "You CANNOT access specific user accounts."
)

# --- HELPER: CONTENT EXTRACTOR ---
def get_text_content(content_or_message):
    if hasattr(content_or_message, 'content'):
        content = content_or_message.content
    else:
        content = content_or_message

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and 'text' in part:
                text_parts.append(part['text'])
            else:
                text_parts.append(str(part))
        return " ".join(text_parts)
    return ""

# --- 3. DEFINE SUPERVISOR (ROUTER) ---
def supervisor_node(state: MessagesState):
    last_message = state['messages'][-1]
    user_input = get_text_content(last_message)
    
    if not user_input.strip(): return {"next": "INFO"} 

    # KEYWORD OVERRIDE
    account_keywords = ["balance", "money", "spent", "spend", "transaction", "checking", "savings"]
    if any(keyword in user_input.lower() for keyword in account_keywords):
        print(f"DEBUG: Keyword override -> ACCOUNT")
        return {"next": "ACCOUNT"}

    # LLM ROUTING
    router_prompt = (
        f"Classify the intent: '{user_input}'.\n"
        "Return ACCOUNT if it requires looking up personal data.\n"
        "Return INFO if it is a general question.\n"
        "Return ONLY the word ACCOUNT or INFO."
    )
    
    try:
        response = llm.invoke(router_prompt)
        choice_text = get_text_content(response)
        choice = choice_text.strip().upper()
        print(f"DEBUG: LLM routed to {choice}")
    except Exception as e:
        print(f"DEBUG: Router Error {e}")
        return {"next": "INFO"}
    
    if "ACCOUNT" in choice: return {"next": "ACCOUNT"}
    return {"next": "INFO"}

# --- 4. WRAPPERS ---
def call_account(state: MessagesState):
    messages = [SystemMessage(content=ACCOUNT_PROMPT)] + state['messages']
    result = account_agent.invoke({"messages": messages})
    return {"messages": [result['messages'][-1]]}

def call_info(state: MessagesState):
    messages = [SystemMessage(content=INFO_PROMPT)] + state['messages']
    result = info_agent.invoke({"messages": messages})
    return {"messages": [result['messages'][-1]]}

# --- 5. BUILD GRAPH ---
class AgentState(MessagesState):
    next: str

workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("account_bot", call_account)
workflow.add_node("info_bot", call_info)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda x: x['next'],
    {"ACCOUNT": "account_bot", "INFO": "info_bot"}
)

workflow.add_edge("account_bot", END)
workflow.add_edge("info_bot", END)

# --- NEW: COMPILE WITH MEMORY ---
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)