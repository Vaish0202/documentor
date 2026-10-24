from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATA_DIR, INDEX_DIR, get_embeddings


def load_documents(folder: str):
    docs = []
    for path in Path(folder).rglob("*"):
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() in {".txt", ".md"}:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    return docs


def main():
    docs = load_documents(DATA_DIR)
    if not docs:
        raise SystemExit(f"No documents found in ./{DATA_DIR}. Add PDFs or .txt/.md files.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} pages/files -> {len(chunks)} chunks")

    FAISS.from_documents(chunks, get_embeddings()).save_local(INDEX_DIR)
    print(f"Index saved to ./{INDEX_DIR}")


if __name__ == "__main__":
    main()
