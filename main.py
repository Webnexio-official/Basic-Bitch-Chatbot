# =========================
# ENV LOADING (LOCAL ONLY)
# =========================
from pathlib import Path
import os

# Only load .env locally
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)

# =========================
# IMPORTS
# =========================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from supabase import create_client
from typing import List, Dict

# =========================
# CONFIG (FROM ENV)
# =========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")

LLM_ENDPOINT = os.getenv(
    "LLM_ENDPOINT",
    "https://openrouter.ai/api/v1/chat/completions"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "meta-llama/llama-3-8b-instruct"
)

# =========================
# FAIL FAST
# =========================
if not SUPABASE_URL:
    raise RuntimeError("❌ SUPABASE_URL missing")
if not SUPABASE_ANON_KEY:
    raise RuntimeError("❌ SUPABASE_ANON_KEY missing")
if not LLM_API_KEY:
    raise RuntimeError("❌ LLM_API_KEY missing")

print("✅ Environment loaded")

# =========================
# APP INIT
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

SYSTEM_PROMPT = """
You are a rude website chatbot.
Be concise, sarcastic, and unprofessional.
"""

# =========================
# SCHEMA
# =========================
class ChatInput(BaseModel):
    session_id: str
    message: str

def to_message_list(history_rows: List[Dict]) -> List[Dict]:
    return [
        {"role": r["role"], "content": r["content"]}
        for r in history_rows
        if r.get("role") and r.get("content")
    ]

# =========================
# ROUTES
# =========================
@app.post("/chat")
def chat(data: ChatInput):
    # 1. Save user message
    supabase.table("messages").insert({
        "session_id": data.session_id,
        "role": "user",
        "content": data.message
    }).execute()

    # 2. Fetch history
    res = (
        supabase.table("messages")
        .select("role, content")
        .eq("session_id", data.session_id)
        .order("created_at", desc=True)
        .limit(6)
        .execute()
    )

    history = list(reversed(res.data or []))

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + to_message_list(history)
        + [{"role": "user", "content": data.message}]
    )

    # 3. Call LLM (OpenRouter)
    payload = {
        "model": LLM_MODEL,
        "messages": messages
    }

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://basic-bitch-chatbot.onrender.com",
        "X-Title": "Basic Bitch Chatbot"
    }

    llm_resp = requests.post(
        LLM_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=30
    )

    if llm_resp.status_code != 200:
        return {
            "error": "LLM error",
            "status": llm_resp.status_code,
            "body": llm_resp.text
        }

    assistant_msg = llm_resp.json()["choices"][0]["message"]["content"]

    # 4. Save assistant message
    supabase.table("messages").insert({
        "session_id": data.session_id,
        "role": "assistant",
        "content": assistant_msg
    }).execute()

    return {"reply": assistant_msg}

@app.get("/")
def root():
    return {"status": "ok", "message": "Chatbot backend running"}
