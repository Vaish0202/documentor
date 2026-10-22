# Sample notes: LangChain basics

LangChain is a framework for building applications with large language models.
LCEL (LangChain Expression Language) lets you compose components with the | operator,
for example: prompt | model | output_parser.

A retriever fetches relevant chunks from a vector store. RAG (retrieval-augmented
generation) passes those chunks to the model as context so answers stay grounded
in your documents.

LangSmith records traces of every chain run so you can debug and evaluate them.
LangServe exposes a chain as a REST API with an interactive playground.
