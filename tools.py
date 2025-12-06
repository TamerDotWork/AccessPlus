import csv
import os
from langchain_core.tools import tool

# Helper to get absolute path so it works regardless of where you run python from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_CSV = os.path.join(BASE_DIR, 'data', 'users.csv')
TXNS_CSV = os.path.join(BASE_DIR, 'data', 'transactions.csv')

# --- ACCOUNT TOOLS ---

def get_user_row(user_id):
    try:
        with open(USERS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['user_id'] == user_id: return row
    except FileNotFoundError:
        return None
    return None

@tool
def get_my_balance() -> str:
    """Check the balance of the logged-in user."""
    user_id = "user_101" 
    user = get_user_row(user_id)
    if user:
        return f"Balance: ${user['balance']} ({user['account_type']})"
    return f"Error: User not found or data/users.csv missing."

@tool
def get_my_transactions() -> list:
    """Get recent spending history."""
    user_id = "user_101"
    txns = []
    try:
        with open(TXNS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['user_id'] == user_id:
                    txns.append(f"{row['date']}: {row['merchant']} (${row['amount']})")
    except FileNotFoundError:
        return ["Error: data/transactions.csv not found"]
    return txns[-5:]

# --- INFO TOOLS ---

@tool
def get_bank_policies(topic: str) -> str:
    """Retrieves general bank information (fees, hours, rates)."""
    policies = {
        "fees": "There is a $5 monthly fee for Checking accounts under $500.",
        "hours": "Branches are open 9am-5pm Mon-Fri.",
        "rates": "Savings APY is currently 4.5%."
    }
    for key, value in policies.items():
        if key in topic.lower():
            return value
    return "I couldn't find a specific policy on that."