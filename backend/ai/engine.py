import os
from datetime import datetime
from typing import Dict, List, Any, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from ingestion.vector_store import get_retriever
from ai.prompts import SYSTEM_PROMPT, URGENCY_PROMPT
from routers.users import DB, UserProfile

# Required env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class AgentState(TypedDict):
    user_id: str
    inquiry: str
    history: List[Dict[str, str]]
    user_profile: Dict[str, Any]
    retrieved_docs: str
    reasoning_steps: List[str]
    response: str
    emergency: bool

# Define nodes for the LangGraph state machine
def load_user_profile(state: AgentState):
    """Loads the user's profile and constraints from the DB"""
    user_id = state["user_id"]
    profile = DB.get(user_id)
    if not profile:
        profile = UserProfile(id=user_id, age=0, gender="Unknown", medical_constraints=[], preferences=[], biometrics={})
    
    state["user_profile"] = profile.dict()
    state["reasoning_steps"].append(f"Loaded profile: {profile.medical_constraints}, {profile.preferences}")
    return state

def get_llm(temperature=0.0):
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("CRITICAL ERROR: GITHUB_TOKEN environment variable not set. Please provide a valid GitHub Models token.")
        
    return ChatOpenAI(
        model="gpt-4o", 
        temperature=temperature,
        api_key=token,
        base_url="https://models.inference.ai.azure.com"
    )

def assess_urgency(state: AgentState):
    """Urgency Assessor Node: Determines if the inquiry requires 911"""
    llm = get_llm(temperature=0.0)
    prompt = PromptTemplate.from_template(URGENCY_PROMPT)
    chain = prompt | llm
    
    result = chain.invoke({"inquiry": state["inquiry"]})
    text = result.content.strip()
    
    if text.startswith("EMERGENCY"):
        state["emergency"] = True
        state["response"] = text
        state["reasoning_steps"].append("CRITICAL URGENCY DETECTED. Bypassing RAG.")
    else:
        state["emergency"] = False
        state["reasoning_steps"].append("Urgency Assessor: SAFE. Proceeding to retrieval.")
    return state

def retrieve_knowledge(state: AgentState):
    """RAG Node: Retrieves medical guidelines from ChromaDB via Github Models Embeddings"""
    retriever = get_retriever()
    docs = retriever.invoke(state["inquiry"])
    doc_text = "\n\n".join([d.page_content for d in docs])
    
    state["retrieved_docs"] = doc_text
    state["reasoning_steps"].append(f"Retrieved {len(docs)} relevant medical chunks from Knowledge Base.")
    return state

def generate_response(state: AgentState):
    """Synthesizer Node: Combines History, User Data, Context, and LLM to answer"""
    llm = get_llm(temperature=0.7)
    
    prof = state["user_profile"]
    context_str = f"User is {prof['age']} years old. Constraints: {prof['medical_constraints']}. Preferences: {prof['preferences']}."
    knowledge_str = f"Medical Guidelines:\n{state['retrieved_docs']}"
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT + "\n\n" + context_str + "\n\n" + knowledge_str),
    ]
    
    # Append History
    for msg in state["history"][-5:]: # Last 5 messages context
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(SystemMessage(content=msg["content"]))
            
    # Append current inquiry
    messages.append(HumanMessage(content=state["inquiry"]))
    
    result = llm.invoke(messages)
    
    state["response"] = result.content
    state["reasoning_steps"].append("Synthesizer structured the final response matching constraints.")
    return state

# Define conditional edges
def route_urgency(state: AgentState):
    if state.get("emergency", False):
        return "end" # Early exit for emergency
    return "retrieve"

# Build the LangGraph
workflow = StateGraph(AgentState)

workflow.add_node("load_profile", load_user_profile)
workflow.add_node("assess_urgency", assess_urgency)
workflow.add_node("retrieve", retrieve_knowledge)
workflow.add_node("generate", generate_response)

workflow.set_entry_point("load_profile")
workflow.add_edge("load_profile", "assess_urgency")
workflow.add_conditional_edges("assess_urgency", route_urgency, {"end": END, "retrieve": "retrieve"})
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile engine
engine = workflow.compile()

async def process_chat_message(user_id: str, inquiry: str, history: List[Dict[str, str]]):
    """Main exported function called by the API"""
    initial_state = {
        "user_id": user_id, 
        "inquiry": inquiry, 
        "history": [h.dict() if hasattr(h, 'dict') else h for h in history], 
        "user_profile": {}, 
        "retrieved_docs": "", 
        "reasoning_steps": [], 
        "response": "", 
        "emergency": False
    }
    
    result = engine.invoke(initial_state)
    return result["response"], result["reasoning_steps"]
