"""Central config: provider switch, model builders, embeddings."""
import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
DATA_DIR = os.getenv("DATA_DIR", "data")


def _build_chat_model(groq_model: str | None = None):
    if PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=groq_model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                        temperature=0)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0)


def get_llm(schema=None):
    """Chat model with retries + fallback.

    If `schema` (a Pydantic class) is given, structured output is applied to each
    model BEFORE adding the fallback, because the fallback wrapper itself does
    not expose with_structured_output.
    """
    primary = _build_chat_model()
    if schema is not None:
        primary = primary.with_structured_output(schema)
    primary = primary.with_retry(stop_after_attempt=2)

    if PROVIDER != "groq":
        return primary

    backup = _build_chat_model(os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b"))
    if schema is not None:
        backup = backup.with_structured_output(schema)
    return primary.with_fallbacks([backup])


def get_embeddings():
    """FastEmbed (small ONNX model) is used locally AND when deployed, because a
    FAISS index must be queried with the same embedding model that built it."""
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
