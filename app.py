from rag import (
    load_pdfs,
    split_documents,
    create_vector_store,
    load_vector_store,
    answer_question
)

vector_store = load_vector_store()

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