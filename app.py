from rag import (
    load_pdfs,
    split_documents,
    create_vector_store
)

docs = load_pdfs()
chunks = split_documents(docs)

print(f"Creating embeddings for {len(chunks)} chunks...")

vector_store = create_vector_store(chunks)

print("Vector store created successfully!")