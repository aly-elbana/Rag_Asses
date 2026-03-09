# Rag_Asses

RAG project that loads PDFs into a vector store and answers questions using Gemini via a Gradio chat interface.

## Structure

- `src/config.py` – Paths, API key, and constants
- `src/ingestion.py` – Load PDFs, chunk text, build embeddings (Chroma)
- `src/rag.py` – Retriever and Gemini answer pipeline
- `src/app.py` – Gradio chat UI
- `Rag_Project_NoteBook.ipynb` – Run setup, ingestion, and app
- `requirements.txt` – Dependencies

## Setup

1. Install dependencies:
```
   pip install -r requirements.txt
```
2. Set your Google API key:

```
   - Local: set `GOOGLE_API_KEY` in your environment or in a `.env` file in the project root.
   - Colab: store the key in Notebook secrets; the code reads it via `src.config`.
```

3. Set your HuggingFace API

## Run

Open `Rag_Project_NoteBook.ipynb` and run the cells in order. Put your PDFs in a folder named `Untitled Folder` in the project root (or set `KNOWLEDGE_BASE` in `src/config.py`). The notebook runs ingestion, then you can test and launch the Gradio app.

To run only the app after ingestion (from project root):

```
python app.py
```
