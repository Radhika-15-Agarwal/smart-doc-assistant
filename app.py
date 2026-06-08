from rag import (
    load_pdfs,
    split_documents,
    create_vector_store,
    answer_question
)

docs = load_pdfs()
chunks = split_documents(docs)

vector_store = create_vector_store(chunks)

question = input("Ask a question: ")

answer, sources = answer_question(
    question,
    vector_store
)

print("\nAnswer:\n")
print(answer)

print("\nSources:")
for s in sources:
    print("-", s)