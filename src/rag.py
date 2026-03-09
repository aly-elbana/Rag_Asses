from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import (
    DB_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    RETRIEVAL_K,
    ensure_api_key,
)

ensure_api_key()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from src.config import get_google_api_key
        if not get_google_api_key():
            raise ValueError(
                "GOOGLE_API_KEY is missing. Add it to .env or set it as an environment variable."
            )
        _llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    return _llm

SYSTEM_PROMPT = """You are a knowledgeable, friendly assistant.
You are chatting with a user.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""


def fetch_context(question: str) -> list[Document]:
    return retriever.invoke(question)


def combined_question(question: str, history: list[dict] | None = None) -> str:
    """Use the whole chat (user + assistant) as context for retrieval."""
    if not history:
        return question
    parts = []
    for m in history:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    if not parts:
        return question
    return "\n".join(parts) + "\n" + question


def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[Document]]:
    history = history or []
    combined = combined_question(question, history)
    try:
        docs = fetch_context(combined)
    except Exception as e:
        docs = []
        context_text = f"(Could not access vector store: {e})"
    else:
        context_text = "\n\n".join(doc.page_content for doc in docs) if docs else "(No documents in the knowledge base yet.)"
    formatted_system_prompt = SYSTEM_PROMPT.format(context=context_text)
    messages = [SystemMessage(content=formatted_system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = _get_llm().invoke(messages)
    content = getattr(response, "content", None) or str(response)
    return content, docs
