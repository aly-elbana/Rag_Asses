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
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

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
    if not history:
        return question
    prior = "\n".join(m["content"] for m in history if m.get("role") == "user")
    return prior + "\n" + question if prior else question


def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[Document]]:
    history = history or []
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context_text = "\n\n".join(doc.page_content for doc in docs)
    formatted_system_prompt = SYSTEM_PROMPT.format(context=context_text)
    messages = [SystemMessage(content=formatted_system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
