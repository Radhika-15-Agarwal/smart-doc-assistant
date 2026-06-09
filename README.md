# Smart PDF Assistant

An agentic PDF question-answering system built using LangGraph, ChromaDB, HuggingFace embeddings, Gemini, and Streamlit.

## Features

* PDF upload support
* Multi-PDF support
* Semantic search with ChromaDB
* Agentic retrieval workflow using LangGraph
* Query rewriting for improved retrieval
* Retrieval quality evaluation
* Gemini-powered answers
* Streamlit web interface

## Tech Stack

* Python
* LangChain
* LangGraph
* ChromaDB
* HuggingFace Embeddings
* Google Gemini
* Streamlit

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

├── streamlit_app.py

├── rag.py

├── graph.py

├── requirements.txt

├── README.md

├── data/

└── chroma_db/

## Future Improvements

* Source citations and page highlighting
* Local LLM support via Ollama
* Conversational memory
* Document summarization mode
