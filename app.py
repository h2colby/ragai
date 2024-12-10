from flask import Flask, request, jsonify, send_from_directory
from rag_pipeline import answer_question

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')
    response = answer_question(question)
    return jsonify({"answer": response})

@app.route('/')
def index():
    # Serve the index.html file from the current directory
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    # Run the Flask server locally
    # Access via http://localhost:5000
    app.run(host='0.0.0.0', port=5000, debug=True)
