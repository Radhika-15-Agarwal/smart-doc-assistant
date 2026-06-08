from rag import load_pdfs

docs = load_pdfs()

print(f"Loaded {len(docs)} pages")