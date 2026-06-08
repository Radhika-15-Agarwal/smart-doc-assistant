from rag import load_vector_store

from graph import build_graph


vector_store = load_vector_store()

graph = build_graph(vector_store)

question = input("Ask a question: ")

result = graph.invoke(
    {
        "question": question
    }
)

print("\nAnswer:\n")
print(result["answer"])