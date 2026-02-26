from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Any
from ai.engine import process_chat_message

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: List[ChatMessage]

class ChatResponse(BaseModel):
    response: str
    reasoning_steps: List[str]

@router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # Connected to the LangGraph AI Engine
    response, reasoning = await process_chat_message(req.user_id, req.message, req.history)
    return ChatResponse(
        response=response,
        reasoning_steps=reasoning
    )
