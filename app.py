from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    # LangChain <=0.2 path
    from langchain.chains import ConversationalRetrievalChain
except ModuleNotFoundError:
    # LangChain >=1.0 keeps classic chains in langchain_classic
    from langchain_classic.chains import ConversationalRetrievalChain

DEFAULT_DATA_PATH = Path("data/medical_guidelines.txt")
DEFAULT_GITHUB_MODELS_ENDPOINT = "https://models.github.ai/inference"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-large"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_RETRIEVAL_K = 4

NOT_ENOUGH_INFO_MESSAGE = (
    "I do not have enough information in my current medical database to answer this"
)
MANDATORY_DISCLAIMER = (
    "Disclaimer: I am an AI, not a doctor. Please consult a healthcare professional "
    "for medical advice."
)


def configure_page() -> None:
    """Configure page layout and static UI text."""
    st.set_page_config(
        page_title="NeuroHealth RAG Assistant",
        page_icon="N",
        layout="centered",
    )
    st.title("NeuroHealth: AI-Powered Health Assistant (PoC)")
    st.caption(
        "This prototype answers only from a local medical guideline dataset using RAG."
    )


def initialize_session_state() -> None:
    """Initialize session state keys used across Streamlit reruns."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def load_medical_documents(file_path: Path) -> list[Document]:
    """Load raw guideline text from the local file into LangChain documents.

    Args:
        file_path: Path to the local medical guideline text file.

    Returns:
        A list of LangChain Document objects.

    Raises:
        FileNotFoundError: If the source file is missing.
        Exception: If parsing/loading fails for any other reason.
    """
    try:
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()
    except FileNotFoundError as exc:
        st.error(
            f"Knowledge base file is missing: `{file_path}`. "
            "Please add `data/medical_guidelines.txt` and rerun."
        )
        raise exc
    except Exception as exc:
        st.error(
            "Unable to load the medical guidelines file. "
            "Please verify the file is readable UTF-8 text."
        )
        raise exc


def split_documents(
    documents: list[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Split source documents into retrieval-friendly chunks.

    Args:
        documents: Raw documents loaded from disk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Character overlap between neighboring chunks.

    Returns:
        Chunked document list for vector indexing.
    """
    # ~700 chars preserves enough triage detail per chunk; overlap keeps
    # sentence continuity when symptom/action statements cross chunk boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> OpenAIEmbeddings:
    """Initialize and cache the embedding model.

    Args:
        model_name: GitHub Models embedding identifier.

    Returns:
        Configured OpenAIEmbeddings instance targeting GitHub Models endpoint.

    Raises:
        ValueError: If `GITHUB_TOKEN` is missing.
    """
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ValueError(
            "GITHUB_TOKEN is not set. Add it to your environment or `.env` file."
        )

    base_url = os.getenv("GITHUB_MODELS_ENDPOINT", DEFAULT_GITHUB_MODELS_ENDPOINT)
    signature = inspect.signature(OpenAIEmbeddings)

    init_kwargs: dict[str, Any] = {}
    if "model" in signature.parameters:
        init_kwargs["model"] = model_name
    elif "model_name" in signature.parameters:
        init_kwargs["model_name"] = model_name

    # Support both modern and legacy argument names across langchain-openai versions.
    if "api_key" in signature.parameters:
        init_kwargs["api_key"] = github_token
    elif "openai_api_key" in signature.parameters:
        init_kwargs["openai_api_key"] = github_token

    if "base_url" in signature.parameters:
        init_kwargs["base_url"] = base_url
    elif "openai_api_base" in signature.parameters:
        init_kwargs["openai_api_base"] = base_url

    return OpenAIEmbeddings(**init_kwargs)


@st.cache_resource(show_spinner="Building local medical FAISS index...")
def build_vector_store(
    file_path: str,
    file_mtime: float,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> FAISS:
    """Create and cache a FAISS vector store from local guideline text.

    Args:
        file_path: String path to the guideline source file.
        file_mtime: File modification timestamp used for cache invalidation.
        embedding_model_name: Embedding model identifier.
        chunk_size: Maximum chunk size for text splitting.
        chunk_overlap: Overlap between chunks to retain context continuity.

    Returns:
        A FAISS vector index initialized from chunked guideline documents.
    """
    # `file_mtime` participates in Streamlit's cache key to auto-rebuild
    # only when the source text file changes.
    _ = file_mtime

    raw_docs = load_medical_documents(Path(file_path))
    chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = get_embeddings(embedding_model_name)
    return FAISS.from_documents(chunks, embeddings)


def initialize_llm(
    model_name: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Initialize the chat model backend.

    Args:
        model_name: OpenAI model identifier (default: `gpt-4o`).
        temperature: Sampling temperature.

    Returns:
        An initialized LangChain chat model.

    Raises:
        ValueError: If required environment variables are missing.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your environment or `.env` file."
        )

    signature = inspect.signature(ChatOpenAI)
    init_kwargs: dict[str, Any] = {"temperature": temperature}

    if "model" in signature.parameters:
        init_kwargs["model"] = model_name
    elif "model_name" in signature.parameters:
        init_kwargs["model_name"] = model_name

    # Support both modern and legacy argument names across langchain-openai versions.
    if "api_key" in signature.parameters:
        init_kwargs["api_key"] = openai_api_key
    elif "openai_api_key" in signature.parameters:
        init_kwargs["openai_api_key"] = openai_api_key

    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if openai_base_url:
        if "base_url" in signature.parameters:
            init_kwargs["base_url"] = openai_base_url
        elif "openai_api_base" in signature.parameters:
            init_kwargs["openai_api_base"] = openai_base_url

    return ChatOpenAI(**init_kwargs)


def build_system_prompt() -> PromptTemplate:
    """Create the strict RAG prompt used by the answer synthesis chain.

    Returns:
        A PromptTemplate enforcing context-only medical responses.
    """
    template = f"""
You are NeuroHealth, a preliminary AI Health Assistant.

Follow these rules exactly for every answer:
1) Use ONLY the retrieved context provided below.
2) If the answer is not present in the context, reply exactly with:
"{NOT_ENOUGH_INFO_MESSAGE}"
3) Do not use external knowledge, assumptions, or speculation.
4) End every response with this exact sentence:
"{MANDATORY_DISCLAIMER}"

Retrieved context:
{{context}}

Conversation history:
{{chat_history}}

User question:
{{question}}

Answer:
""".strip()

    return PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"],
    )


def build_rag_chain(
    llm: BaseChatModel,
    vector_store: FAISS,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
) -> ConversationalRetrievalChain:
    """Build the conversational retrieval chain.

    Args:
        llm: Chat model used for answer generation.
        vector_store: FAISS vector index containing guideline chunks.
        retrieval_k: Number of top chunks retrieved per user query.

    Returns:
        A configured ConversationalRetrievalChain instance.
    """
    # k=4 gives enough recall for short triage records while limiting
    # irrelevant context and prompt size.
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retrieval_k},
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": build_system_prompt()},
    )


def to_chat_history_pairs(messages: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Convert flat message history into user-assistant pairs for LangChain.

    Args:
        messages: Streamlit message objects in chronological order.

    Returns:
        List of `(user_message, assistant_message)` tuples.
    """
    history: list[tuple[str, str]] = []
    pending_user_message: str | None = None

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            pending_user_message = content
        elif role == "assistant" and pending_user_message is not None:
            history.append((pending_user_message, content))
            pending_user_message = None

    return history


def enforce_disclaimer(response_text: str) -> str:
    """Ensure the mandatory disclaimer is always included.

    Args:
        response_text: LLM-produced answer text.

    Returns:
        Answer text guaranteed to include the required disclaimer.
    """
    normalized = response_text.strip() or NOT_ENOUGH_INFO_MESSAGE
    if MANDATORY_DISCLAIMER not in normalized:
        return f"{normalized}\n\n{MANDATORY_DISCLAIMER}"
    return normalized


def generate_response(
    chain: ConversationalRetrievalChain,
    user_question: str,
    chat_history: list[tuple[str, str]],
) -> str:
    """Invoke the RAG chain and return a safe assistant response.

    Args:
        chain: Conversational retrieval chain.
        user_question: Current user question.
        chat_history: Previous conversation turns as user-assistant tuples.

    Returns:
        Assistant response text with mandatory disclaimer.
    """
    try:
        result: dict[str, Any] = chain.invoke(
            {"question": user_question, "chat_history": chat_history}
        )
        answer = str(result.get("answer", "")).strip()
        if not answer:
            answer = NOT_ENOUGH_INFO_MESSAGE
        return enforce_disclaimer(answer)
    except Exception:
        st.error(
            "The LLM request failed. Please verify your API key/provider setup and try again."
        )
        return enforce_disclaimer(NOT_ENOUGH_INFO_MESSAGE)


def render_sidebar() -> tuple[str, str, int]:
    """Render app controls in the sidebar.

    Returns:
        Tuple of LLM model, embedding model, and retrieval top-k.
    """
    with st.sidebar:
        st.header("Configuration")
        llm_model_name = st.text_input(
            "OpenAI Model",
            value=DEFAULT_OPENAI_MODEL,
            help="Set to `gpt-4o` by default. Requires OPENAI_API_KEY.",
        )
        embedding_model_name = st.text_input(
            "GitHub Embedding Model",
            value=DEFAULT_EMBEDDING_MODEL,
            help="Uses GITHUB_TOKEN with GitHub Models inference endpoint.",
        )
        st.caption("Required: `OPENAI_API_KEY` and `GITHUB_TOKEN`.")

        retrieval_k = st.slider("Retriever top-k", min_value=2, max_value=8, value=4)
        st.markdown("---")
        st.write("Knowledge Base: `data/medical_guidelines.txt`")

    return llm_model_name, embedding_model_name, retrieval_k


def render_chat_history(messages: list[dict[str, str]]) -> None:
    """Render existing chat history from session state.

    Args:
        messages: Stored user/assistant chat messages.

    Returns:
        None.
    """
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def validate_data_path_or_stop(data_path: Path) -> None:
    """Validate the local knowledge base file and stop app on failure.

    Args:
        data_path: Path to the guideline source file.
    """
    if not data_path.exists():
        st.error(
            "Medical knowledge file is missing at `data/medical_guidelines.txt`. "
            "Please add the file and rerun the app."
        )
        st.stop()


def run_app() -> None:
    """Run the Streamlit chatbot application flow."""
    configure_page()
    initialize_session_state()

    llm_model_name, embedding_model_name, retrieval_k = render_sidebar()
    validate_data_path_or_stop(DEFAULT_DATA_PATH)
    render_chat_history(st.session_state.messages)

    user_question = st.chat_input("Describe your symptoms or ask a triage question...")
    if not user_question:
        return

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    prior_history = to_chat_history_pairs(st.session_state.messages[:-1])
    with st.chat_message("assistant"):
        with st.spinner("Reviewing local medical guidance..."):
            try:
                vector_store = build_vector_store(
                    file_path=str(DEFAULT_DATA_PATH),
                    file_mtime=DEFAULT_DATA_PATH.stat().st_mtime,
                    embedding_model_name=embedding_model_name,
                )
                llm = initialize_llm(model_name=llm_model_name, temperature=0.0)
                rag_chain = build_rag_chain(
                    llm=llm,
                    vector_store=vector_store,
                    retrieval_k=retrieval_k,
                )
                assistant_response = generate_response(
                    chain=rag_chain,
                    user_question=user_question,
                    chat_history=prior_history,
                )
            except Exception as exc:
                st.error(f"Initialization error: {exc}")
                assistant_response = enforce_disclaimer(NOT_ENOUGH_INFO_MESSAGE)
        st.markdown(assistant_response)

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})


def main() -> None:
    """Entry point for Streamlit execution."""
    load_dotenv()
    run_app()


if __name__ == "__main__":
    main()
