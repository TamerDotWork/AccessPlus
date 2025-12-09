import os
import sys
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, START, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver 
from langsmith import traceable


from tools import get_my_balance, get_my_transactions, get_bank_policies

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY is missing from .env file.")
    sys.exit(1)

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    temperature=0
)


class GuardianOutput(BaseModel):
    is_allowed: bool = Field(..., description="True if banking related, False otherwise.")
    reason: str = Field(..., description="Reason for decision.")

class RouterOutput(BaseModel):
    destination: Literal["account_bot", "info_bot"] = Field(..., description="Target agent.")


account_agent = create_react_agent(llm, tools=[get_my_balance, get_my_transactions])
info_agent = create_react_agent(llm, tools=[get_bank_policies])

ACCOUNT_SYS = "You are an Account Manager. Use tools for balances/transactions. Refuse non-banking queries."
INFO_SYS = "You are a Bank Consultant. Answer fees/rates/hours. You CANNOT access accounts. Refuse non-banking queries."

@traceable
def guardian_node(state: MessagesState):
    """Firewall Node with Crash Prevention."""
    last_msg = state['messages'][-1]
    
    prompt = (
        "You are a Banking Gateway Security System. Your job is to filter user messages."
        "Analyze the user's input and decide if it is related to banking, finance, or account management."
        "CRITERIA FOR 'is_allowed' (True):"
        "- Account inquiries (e.g., 'my balance', 'how much money', 'transactions')."
        "- General banking questions (e.g., 'fees', 'interest rates', 'hours')."
        "- Greetings (e.g., 'hi', 'hello')."
        "CRITERIA FOR 'is_allowed' (False):"
        "- General knowledge questions (e.g., 'capital of France', 'python code')."
        "- Creative writing, recipes, or personal advice unrelated to money."
        "Provide a clear 'reason' for your decision."
    )
    
    try:
        structured_llm = llm.with_structured_output(GuardianOutput)
        decision = structured_llm.invoke([SystemMessage(content=prompt), last_msg])
    except Exception as e:
        print(f"Guardian Error: {e}")
        decision = None

    if decision is None:
        print("Guardian returned None. Defaulting to BLOCK.")
        decision = GuardianOutput(is_allowed=False, reason="System safety check failed.")
    
    return {"guardian_decision": decision}
@traceable
def router_node(state: MessagesState):
    """Router Node with Crash Prevention."""
    messages = state['messages']
    
    prompt = (
        "You are a Banking Router. Route the user message to the correct specialist agent."
        "DESTINATION RULES:"
        "1. 'account_bot': STRICTLY for personal/private account data."
        "   - Use for: 'my balance', 'my transactions', 'did I spend money at X', 'transfer money'."
        "   - The user is asking about THEIR specific money."
        "2. 'info_bot': STRICTLY for general/public bank policies."
        "   - Use for: 'what are the fees', 'interest rates', 'opening hours', 'how do I open an account'."
        "   - The user is asking about the BANK, not their specific money."
    )
    
    try:
        structured_llm = llm.with_structured_output(RouterOutput)
        decision = structured_llm.invoke([SystemMessage(content=prompt)] + messages)
    except Exception as e:
        print(f"Router Error: {e}")
        decision = None

    if decision is None:
        print("Router returned None. Defaulting to info_bot.")
        return {"next": "info_bot"}
    
    print(f"🚦 Routing to: {decision.destination}")
    return {"next": decision.destination}
@traceable
def call_account(state: MessagesState):
    msg = [SystemMessage(content=ACCOUNT_SYS)] + state['messages']
    res = account_agent.invoke({"messages": msg})
    return {"messages": [res['messages'][-1]]}
@traceable
def call_info(state: MessagesState):
    msg = [SystemMessage(content=INFO_SYS)] + state['messages']
    res = info_agent.invoke({"messages": msg})
    return {"messages": [res['messages'][-1]]}
@traceable
def call_block(state: MessagesState):
    return {"messages": [AIMessage(content="I cannot assist with that request. Please ask about banking.")]}


class AgentState(MessagesState):
    guardian_decision: GuardianOutput
    next: str

workflow = StateGraph(AgentState)

workflow.add_node("guardian", guardian_node)
workflow.add_node("router", router_node)
workflow.add_node("account_bot", call_account)
workflow.add_node("info_bot", call_info)
workflow.add_node("block_bot", call_block)

workflow.set_entry_point("guardian")
@traceable
def route_guardian(state: AgentState):
    if state.get("guardian_decision") and state["guardian_decision"].is_allowed:
        return "router"
    return "block_bot"

workflow.add_conditional_edges("guardian", route_guardian, ["router", "block_bot"])
workflow.add_conditional_edges("router", lambda x: x['next'], ["account_bot", "info_bot"])

workflow.add_edge("account_bot", END)
workflow.add_edge("info_bot", END)
workflow.add_edge("block_bot", END)

memory = MemorySaver()


app_graph = workflow.compile(checkpointer=memory)