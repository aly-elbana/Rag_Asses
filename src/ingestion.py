from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import KNOWLEDGE_BASE, DB_NAME, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def fetch_documents():
    kb_path = Path(KNOWLEDGE_BASE)
    if not kb_path.exists():
        print(f"Error: {KNOWLEDGE_BASE} not found")
        return []

    pdf_paths = list(kb_path.rglob("*.pdf"))
    documents = []
    print(f"Found {len(pdf_paths)} PDFs. Processing")

    for pdf_path in pdf_paths:
        try:
            doc_type = pdf_path.parent.name
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for page in pages:
                page.metadata["doc_type"] = doc_type
                page.metadata["source_file"] = pdf_path.name
                documents.append(page)
            print(f"Loaded: {pdf_path.name}")
        except Exception as e:
            print(f"Skipping {pdf_path.name}: {e}")

    return documents


def create_chunks(documents):
    if not documents:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)


def create_embeddings(chunks):
    if not chunks:
        print("Empty chunks list. Nothing to embed")
        return None
    print(f"Generating embeddings for {len(chunks)} chunks")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME
    )
    print(f"Ingestion complete. DB stored at: {DB_NAME}")
    return vectorstore


def run_ingestion():
    docs = fetch_documents()
    if not docs:
        print("No documents found. Check knowledge base folder")
        return None
    chunks = create_chunks(docs)
    return create_embeddings(chunks)
