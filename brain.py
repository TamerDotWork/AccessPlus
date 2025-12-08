# State-of-the-art multi-agent banking assistant with strict guardrails
# Agents: AccountAgent, InfoAgent, ComplianceGuardAgent, Supervisor
# Features: Pre-LLM filtering, post-LLM filtering, prompt injection detection, strict professional fallback, multi-step guidance

import os
import sys
import re
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from tools import get_my_balance, get_my_transactions, get_bank_policies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("banking_agent")

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    logger.error("GOOGLE_API_KEY is missing from .env file.")
    sys.exit(1)

api_key = os.environ.get("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

# --- System prompts with guardrails ---
ACCOUNT_PROMPT = (
    "You are a professional Banking Assistant in a DEMO environment.\n"
    "Only use the tools get_my_balance and get_my_transactions.\n"
    "Do not provide any non-banking advice.\n"
    "Always provide multi-step options if data is missing."
)

INFO_PROMPT = (
    "You are a professional Bank Consultant. Answer questions about fees, rates, and policies.\n"
    "Do not access user account information.\n"
    "Never answer non-banking requests."
)

COMPLIANCE_PROMPT = (
    "You are a Compliance Guard Agent. Detect off-topic, unsafe, or risky queries.\n"
    "Block any request unrelated to banking/account services.\n"
    "Return only professional fallback instructions if blocked."
)

ROUTER_PROMPT_TEMPLATE = (
    "Classify user intent strictly.\n"
    "User input: '{user_input}'\n"
    "Return exactly one of: ACCOUNT, INFO, OFF_TOPIC. Nothing else."
)

# --- Pre-processing and guardrails ---
MAX_INPUT_LEN = 2000
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")
INJECTION_PATTERNS = [
    r"ignore (previous|above).*instructions",
    r"disregard (previous|above).*instructions",
    r"forget (your|the) instructions",
    r"follow these new instructions",
    r"<script>",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)
PII_RE = re.compile(r"\b(?:\d{12,19}|\d{3}-\d{2}-\d{4}|\d{9})\b")
PROHIBITED_RESPONSE_PATTERNS = [r"open the database", r"delete records", r"expose.*credentials"]
PROHIBITED_RE = re.compile("|".join(PROHIBITED_RESPONSE_PATTERNS), re.IGNORECASE)
OFF_TOPIC_KEYWORDS = [
    "joke", "funny", "story", "meme", "laugh", "riddle",
    "travel", "passport", "vacation", "holiday", "flight", "visa",
    "recipe", "sports", "weather", "movie"
]

# --- Helpers ---
def sanitize_user_input(text: str) -> str:
    if not isinstance(text, str): return ""
    text = CONTROL_CHAR_RE.sub("", text).strip()
    return text[:MAX_INPUT_LEN]

def detect_prompt_injection(text: str) -> bool:
    return bool(text and INJECTION_RE.search(text))

def redact_sensitive_info(text: str) -> str:
    if not text: return text
    return PII_RE.sub("[REDACTED]", text)

def post_process_response(response: str) -> str:
    response = redact_sensitive_info(response)
    if PROHIBITED_RE.search(response):
        return (
            "I'm unable to fulfill that request. Please choose one of the professional options:\n"
            "1. Contact human support\n2. Retry with a banking query\n3. View FAQs"
        )
    return response

def get_text_content(content_or_message):
    if hasattr(content_or_message, 'content'):
        content = content_or_message.content
    else:
        content = content_or_message
    if isinstance(content, str): return content
    if isinstance(content, list):
        return ' '.join([part['text'] if isinstance(part, dict) and 'text' in part else str(part) for part in content])
    return ''

# --- Agents ---
account_agent = create_react_agent(llm, tools=[get_my_balance, get_my_transactions])
info_agent = create_react_agent(llm, tools=[get_bank_policies])
compliance_agent = create_react_agent(llm, tools=[])  # No tools needed, empty list

# --- Supervisor / Router ---
def supervisor_node(state: MessagesState):
    last_message = state['messages'][-1]
    user_input = sanitize_user_input(get_text_content(last_message))

    if not user_input:
        return {"next": "INFO"}

    # Pre-LLM Compliance Guard
    if detect_prompt_injection(user_input) or any(k in user_input.lower() for k in OFF_TOPIC_KEYWORDS):
        fallback_msg = SystemMessage(
            content="I am a banking assistant and cannot answer off-topic or unsafe requests."
                    " Please choose one of the following professional options:\n"
                    "1. Check account balance\n2. View transactions\n3. Ask about bank policies or fees\n4. Contact human support"
        )
        return {"next": "INFO", "messages": [fallback_msg]}

    # Account keyword routing
    account_keywords = ["balance", "money", "spent", "spend", "transaction", "checking", "savings", "transfer", "pay", "deposit", "withdraw", "spendings"]
    if any(keyword in user_input.lower() for keyword in account_keywords):
        return {"next": "ACCOUNT"}

    # LLM fallback router
    router_prompt = ROUTER_PROMPT_TEMPLATE.format(user_input=user_input)
    try:
        choice_text = get_text_content(llm.invoke(router_prompt))
        choice = choice_text.strip().upper()
    except Exception:
        choice = "INFO"

    if "ACCOUNT" in choice: return {"next": "ACCOUNT"}
    return {"next": "INFO"}

# --- Agent Calls ---
def call_account(state: MessagesState):
    messages = [SystemMessage(content=ACCOUNT_PROMPT)] + state['messages']
    result = account_agent.invoke({"messages": messages})
    final = post_process_response(get_text_content(result['messages'][-1]))
    return {"messages": [SystemMessage(content=final)]}

def call_info(state: MessagesState):
    messages = [SystemMessage(content=INFO_PROMPT)] + state['messages']
    result = info_agent.invoke({"messages": messages})
    final = post_process_response(get_text_content(result['messages'][-1]))
    return {"messages": [SystemMessage(content=final)]}

# --- State Graph ---
class AgentState(MessagesState):
    next: str

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("account_bot", call_account)
workflow.add_node("info_bot", call_info)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", lambda x: x.get('next'), {"ACCOUNT": "account_bot", "INFO": "info_bot"})
workflow.add_edge("account_bot", END)
workflow.add_edge("info_bot", END)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)