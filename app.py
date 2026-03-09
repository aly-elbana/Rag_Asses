import gradio as gr
from pathlib import Path

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
    formatted_history = _format_history(history)
    answer, docs = answer_question(message, formatted_history)
    unique_sources = set()
    for doc in docs:
        source_name = doc.metadata.get("source", "Unknown Document")
        unique_sources.add(Path(str(source_name)).name)
    source_list = "\n\n---\n**Sources used:**"
    for s in unique_sources:
        source_list += f"\n- {s}"
    return f"{answer}{source_list}"


def create_demo():
    return gr.ChatInterface(
        fn=chat_wrapper,
        title="Gemini RAG Assistant",
        description="Ask questions about your documents.",
    )


def launch(share=False, debug=True):
    demo = create_demo()
    demo.launch(share=share, debug=debug)
