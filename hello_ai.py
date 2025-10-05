import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)
_client = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        _client = OpenAI(api_key=api_key)
    return _client


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "A non-empty 'message' field is required."}), 400

    try:
        client = get_openai_client()
        response = client.responses.create(
            model="gpt-4o-mini",
            input=message,
        )
    except Exception as exc:
        return jsonify({"error": f"Failed to reach OpenAI API: {exc}"}), 500

    reply_text = (response.output_text or "").strip()
    if not reply_text:
        reply_text = "No response received from the AI."

    return jsonify({"reply": reply_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
