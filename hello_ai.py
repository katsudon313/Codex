import json
import os
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()

import firebase_admin
from firebase_admin import credentials
import requests
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from openai import OpenAI

APP_ROOT = Path(__file__).resolve().parent
FIREBASE_CONFIG_PATH = APP_ROOT / "firebase_config.json"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-me-with-a-secure-secret")

_openai_client: OpenAI | None = None
_firebase_initialized = False


@lru_cache(maxsize=1)
def get_firebase_config() -> Dict[str, Any]:
    if not FIREBASE_CONFIG_PATH.exists():
        raise RuntimeError(
            f"Firebase config file not found at {FIREBASE_CONFIG_PATH}."
        )
    with FIREBASE_CONFIG_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


@lru_cache(maxsize=1)
def get_firebase_api_key() -> str:
    config = get_firebase_config()
    try:
        key = config["apiKey"]
    except KeyError as exc:
        raise RuntimeError(
            "Firebase API key missing in firebase_config.json (expected top-level 'apiKey')."
        ) from exc

    if not isinstance(key, str) or not key.strip():
        raise RuntimeError(
            "Firebase API key in firebase_config.json must be a non-empty string under 'apiKey'."
        )

    return key.strip()


def _load_service_account(config: Dict[str, Any]) -> Dict[str, Any]:
    if config.get("type") == "service_account":
        return config

    service_account = config.get("serviceAccount")
    if isinstance(service_account, dict):
        return service_account

    if isinstance(service_account, str):
        candidate = Path(service_account)
        if not candidate.is_absolute():
            candidate = FIREBASE_CONFIG_PATH.parent / candidate
        with candidate.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    raise RuntimeError(
        "Service account credentials not found in firebase_config.json."
    )


def ensure_firebase_initialized() -> None:
    global _firebase_initialized
    if _firebase_initialized:
        return

    config = get_firebase_config()
    creds = credentials.Certificate(_load_service_account(config))
    firebase_admin.initialize_app(creds)
    _firebase_initialized = True


def authenticate_with_firebase(email: str, password: str) -> Dict[str, Any]:
    ensure_firebase_initialized()
    api_key = get_firebase_api_key()
    endpoint = (
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
    )
    try:
        response = requests.post(
            f"{endpoint}?key={api_key}",
            json={
                "email": email,
                "password": password,
                "returnSecureToken": True,
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        try:
            payload = response.json()
            message = payload.get("error", {}).get("message", "Authentication failed.")
        except Exception:  # pragma: no cover - best effort logging only
            message = "Authentication failed."
        raise RuntimeError(message) from http_err
    except requests.RequestException as req_err:
        raise RuntimeError(f"Could not reach Firebase Authentication: {req_err}") from req_err


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY") or "YOUR_OPENAI_API_KEY_HERE"
        if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. Update with a valid key."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_email" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


@app.route("/")
def root():
    if "user_email" in session:
        return redirect(url_for("chat"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""

        if not email or not password:
            error = "Email and password are required."
        else:
            try:
                auth_result = authenticate_with_firebase(email, password)
                session.clear()
                session["user_email"] = auth_result.get("email", email)
                session["id_token"] = auth_result.get("idToken")
                return redirect(url_for("chat"))
            except RuntimeError as exc:
                error = str(exc)

    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/chat", methods=["GET", "POST"])
@login_required
def chat():
    if request.method == "GET":
        return render_template("chat.html", user_email=session.get("user_email"))

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
