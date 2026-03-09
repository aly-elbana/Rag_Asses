import sys
from pathlib import Path

# Ensure project root is on path so "python src/app.py" works from repo root
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Build chunks and vector DB on every run (before loading RAG)
print("Building knowledge base (chunking PDFs and creating embeddings)...")
from src.ingestion import run_ingestion
run_ingestion()
print("Ready. Launching interface...")

import gradio as gr

from src.rag import answer_question


def _format_history(history):
    formatted = []
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            human, ai = item[0], item[1]
            if human:
                formatted.append({"role": "user", "content": str(human)})
            if ai:
                clean_ai = str(ai).split("\n\n---")[0]
                formatted.append({"role": "assistant", "content": clean_ai})
        elif isinstance(item, dict) and "role" in item and "content" in item:
            role = item["role"]
            content = str(item["content"])
            if role == "assistant" and "\n\n---" in content:
                content = content.split("\n\n---")[0]
            formatted.append({"role": role, "content": content})
        elif hasattr(item, "role") and hasattr(item, "content"):
            role = item.role
            content = str(item.content)
            if role == "assistant" and "\n\n---" in content:
                content = content.split("\n\n---")[0]
            formatted.append({"role": role, "content": content})
    return formatted


def chat_wrapper(message, history):
    if not message or not str(message).strip():
        return ""
    try:
        formatted_history = _format_history(history)
        answer, docs = answer_question(message, formatted_history)
        if answer is None:
            return "Error: The model returned no response. Check GOOGLE_API_KEY in .env"
        unique_sources = set()
        for doc in docs:
            source_name = doc.metadata.get("source", "Unknown Document")
            unique_sources.add(Path(str(source_name)).name)
        source_list = "\n\n---\n**Sources used:**"
        for s in unique_sources:
            source_list += f"\n- {s}"
        return f"{answer}{source_list}"
    except Exception as e:
        return f"Error: {e}\n\nMake sure GOOGLE_API_KEY is set in .env and the PDFs folder contains PDF files."


def create_demo():
    return gr.ChatInterface(
        fn=chat_wrapper,
        title="Gemini RAG Assistant",
        description="Ask questions about your documents.",
    )


def launch(share=False, debug=True):
    demo = create_demo()
    demo.launch(share=share, debug=debug)


if __name__ == "__main__":
    launch()
