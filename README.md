# AccessPlus

AccessPlus is an **agentic AI virtual assistant for banking**. It leverages multiple AI agents, retrieval-augmented generation (RAG), and LLMs to provide secure, intelligent, and context-aware banking support via chat.

---

## Features

* Multi-agent architecture (account, info, supervisor router)
* Retrieval-Augmented Generation (RAG) with CSV/FAISS knowledge bases
* LLM-powered natural language understanding and response generation
* Secure banking tools (`get_my_balance`, `get_my_transactions`, `get_bank_policies`)
* Stateful conversations with memory
* FastAPI/Flask web interface for real-time chat
* AI safety and observability

---

## Architecture

```
User Chat UI
      |
Supervisor Router
  |           |
Account      Info
 Agent       Agent
  |           |
Banking Tools / Knowledge Base
      |
      LLM
      |
 Response Back
```

---

## Installation

```bash
git clone https://github.com/TamerDotWork/AccessPlus.git
cd AccessPlus
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
python main.py
```

---

## Usage

```python
from accessplus import AccessPlus

assistant = AccessPlus()
response = assistant.chat("Show my last 5 transactions")
print(response)
```

**Expected Output:**

```
1. - $50 Starbucks
2. - $120 Amazon
3. - $200 Salary Deposit
4. - $15 Netflix
5. - $30 Grocery Store
```

---

## Extending AccessPlus

* Add new agents and register them in the supervisor router
* Integrate additional tools or APIs
* Swap LLMs (Gemini-Flash, Llama, Mistral, etc.)

---

## License

MIT License
