from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load Hugging Face summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.route("/")
def index():
    return "✅ ML Summarizer is running!"

@app.route("/api/summarize", methods=["POST"])
def summarize_text():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    summary = summarizer(text, max_length=120, min_length=30, do_sample=False)
    return jsonify({"summary": summary[0]["summary_text"]})

if __name__ == "__main__":
    # For Railway, listen on all interfaces and use PORT env variable if available
    import os
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)
