from flask import Flask, render_template, request, jsonify
from langchain_core.messages import HumanMessage
from brain import app_graph 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_data = request.json
    user_message = user_data.get('message', '')

    # Guard: Prevent empty strings sending to Gemini
    if not user_message.strip():
        return jsonify({"response": "Please type a valid message."})

    try:
        # Create the HumanMessage object
        input_msg = HumanMessage(content=user_message)
        
        # Invoke Graph
        result = app_graph.invoke({"messages": [input_msg]})
        
        # Get last message
        final_response = result['messages'][-1].content
        
        return jsonify({"response": final_response})
        
    except Exception as e:
        print(f"Server Error: {e}") # Look at your terminal for details
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)