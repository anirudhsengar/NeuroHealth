import os
from dotenv import load_dotenv

# Load env variables before importing routers
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chat, users

app = FastAPI(
    title="NeuroHealth API",
    description="Backend API for the NeuroHealth AI-Powered Health Assistant",
    version="0.1.0",
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(users.router, prefix="/api/users", tags=["users"])

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "NeuroHealth API is running"}
