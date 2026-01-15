import os
import time
import requests
from typing import List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client

# =========================
# CONFIG
# =========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")

LLM_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

PRIMARY_MODEL = "meta-llama/llama-3-8b-instruct"
FALLBACK_MODEL = "mistralai/mistral-7b-instruct"

REQUEST_TIMEOUT = 30  # seconds per model

# =========================
# FAIL FAST
# =========================
if not SUPABASE_URL or not SUPABASE_ANON_KEY or not LLM_API_KEY:
    raise RuntimeError("❌ Missing required environment variables")

print("✅ Environment loaded")

# =========================
# APP INIT
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

SYSTEM_PROMPT = (
    "You are a rude website chatbot. "
    "Be concise, sarcastic, and unfiltered."
)

# =========================
# SCHEMA
# =========================
class ChatInput(BaseModel):
    session_id: str
    message: str


def format_history(rows: List[Dict]) -> List[Dict]:
    return [
        {"role": r["role"], "content": r["content"]}
        for r in rows
        if r.get("role") and r.get("content")
    ]


def call_llm(model: str, messages: List[Dict], logs: List[str]):
    logs.append(f"→ Calling model: {model}")

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://basic-bitch-chatbot.onrender.com",
        "X-Title": "Basic Bitch Chatbot",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    start = time.time()

    resp = requests.post(
        LLM_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    duration = round(time.time() - start, 2)
    logs.append(f"→ Response time: {duration}s")

    if resp.status_code != 200:
        raise RuntimeError(resp.text)

    return resp.json()["choices"][0]["message"]["content"]


# =========================
# ROUTES
# =========================
@app.post("/chat")
def chat(data: ChatInput):
    logs = []
    logs.append("✓ Received user message")

    # Save user message
    supabase.table("messages").insert({
        "session_id": data.session_id,
        "role": "user",
        "content": data.message
    }).execute()
    logs.append("✓ User message saved")

    # Fetch history
    res = (
        supabase.table("messages")
        .select("role, content")
        .eq("session_id", data.session_id)
        .order("created_at", desc=True)
        .limit(6)
        .execute()
    )

    history = list(reversed(res.data or []))
    logs.append(f"✓ Loaded {len(history)} history messages")

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + format_history(history)
        + [{"role": "user", "content": data.message}]
    )

    # Try primary → fallback
    try:
        reply = call_llm(PRIMARY_MODEL, messages, logs)
        used_model = PRIMARY_MODEL
    except Exception as e:
        logs.append("⚠ Primary model failed, switching fallback")
        try:
            reply = call_llm(FALLBACK_MODEL, messages, logs)
            used_model = FALLBACK_MODEL
        except Exception as e2:
            logs.append("❌ Fallback model failed")
            return {
                "error": "LLM failed",
                "logs": logs,
            }

    logs.append(f"✓ Reply generated via {used_model}")

    # Save assistant message
    supabase.table("messages").insert({
        "session_id": data.session_id,
        "role": "assistant",
        "content": reply
    }).execute()

    logs.append("✓ Assistant message saved")

    return {
        "reply": reply,
        "logs": logs,
        "model": used_model,
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "Chatbot backend running"}
