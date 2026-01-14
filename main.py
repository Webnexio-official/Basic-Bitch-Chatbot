# =========================
# IMPORTS
# =========================
import os
import requests
from typing import List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client

# =========================
# CONFIG (RENDER ENV ONLY)
# =========================
# IMPORTANT: DO NOT use dotenv on Render

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]
LLM_API_KEY = os.environ["LLM_API_KEY"]

LLM_ENDPOINT = os.environ.get(
    "LLM_ENDPOINT",
    "https://openrouter.ai/api/v1/chat/completions",
)

LLM_MODEL = os.environ.get(
    "LLM_MODEL",
    "meta-llama/llama-3-8b-instruct",
)

print("✅ Environment variables loaded correctly")

# =========================
# APP INIT
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

SYSTEM_PROMPT = (
    "You are a rude website chatbot. "
    "Be concise, horny, and unprofessional."
)

# =========================
# SCHEMA
# =========================
class ChatInput(BaseModel):
    session_id: str
    message: str


def to_message_list(rows: List[Dict]) -> List[Dict]:
    return [
        {"role": r["role"], "content": r["content"]}
        for r in rows
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

    # 3. OpenRouter request
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://basic-bitch-chatbot.onrender.com",
        "X-Title": "Basic Bitch Chatbot",
    }

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
    }

    llm_resp = requests.post(
        LLM_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=30,
    )

    if llm_resp.status_code != 200:
        return {
            "error": "LLM error",
            "status": llm_resp.status_code,
            "body": llm_resp.text,
        }

    assistant_msg = llm_resp.json()["choices"][0]["message"]["content"]

    # 4. Save assistant reply
    supabase.table("messages").insert({
        "session_id": data.session_id,
        "role": "assistant",
        "content": assistant_msg
    }).execute()

    return {"reply": assistant_msg}


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Chatbot backend running"
    }
