"""LCEL chains: grounded Q&A and structured quiz generation."""
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import INDEX_DIR, get_embeddings, get_llm
from schemas import Quiz

vectorstore = FAISS.load_local(INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 12})


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('source', 'doc')} | p{d.metadata.get('page', '-')}]\n{d.page_content}"
        for d in docs
    )


# ---- 1. Grounded Q&A ------------------------------------------------------
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a study assistant. Answer ONLY from the context. "
     "If the answer is not in the context, say you could not find it in the documents. "
     "Cite sources like [file | page]."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | qa_prompt
    | get_llm()
    | StrOutputParser()
).with_types(input_type=str, output_type=str)


# ---- 2. Structured quiz ----------------------------------------------------
quiz_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Create a 5-question multiple-choice quiz using ONLY the context. "
     "Each question has exactly 4 options and one correct answer."),
    ("human", "Topic: {topic}\n\nContext:\n{context}"),
])

quiz_chain = (
    {"context": retriever | format_docs, "topic": RunnablePassthrough()}
    | quiz_prompt
    | get_llm(schema=Quiz)
).with_types(input_type=str, output_type=Quiz)
