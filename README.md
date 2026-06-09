# Smart PDF Assistant

An agentic PDF question-answering system built using LangGraph, ChromaDB, HuggingFace embeddings, and Gemini.

## Features

* Load and process PDF documents
* Semantic search using vector embeddings
* Retrieval-Augmented Generation (RAG)
* Query rewriting for improved retrieval
* Retrieval quality evaluation
* Source-aware answers

## Tech Stack

* Python
* LangChain
* LangGraph
* ChromaDB
* HuggingFace Embeddings
* Google Gemini

## Architecture

PDF Documents

↓

Chunking

↓

Embeddings

↓

Chroma Vector Store

↓

Retriever

↓

LangGraph Workflow

* Retrieve
* Grade Context
* Rewrite Query (if needed)
* Generate Answer

↓

Gemini

## Project Structure

smart-doc-assistant/

├── app.py

├── rag.py

├── graph.py

├── requirements.txt

├── README.md

├── data/

└── chroma_db/

## Example Questions

* What is the objective of the project?
* Who are the members of Group 1?
* What problem does the project solve?
* What are the implementation phases?

## Future Improvements

* Streamlit UI
* Multi-PDF support
* Local LLM support via Ollama
* Better retrieval grading
* Citation highlighting
