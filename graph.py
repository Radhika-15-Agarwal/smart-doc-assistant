from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from rag import (
    retrieve_documents,
    build_context,
    generate_answer
)


class GraphState(TypedDict):
    question: str
    documents: list
    context: str
    answer: str


def retrieve_node(state: GraphState):

    docs = retrieve_documents(
        state["question"],
        vector_store
    )

    context = build_context(docs)

    return {
        "documents": docs,
        "context": context
    }


def answer_node(state: GraphState):

    answer = generate_answer(
        state["question"],
        state["context"]
    )

    return {
        "answer": answer
    }

def build_graph(vector_store):

    def retrieve(state):

        docs = retrieve_documents(
            state["question"],
            vector_store
        )

        context = build_context(docs)

        return {
            "documents": docs,
            "context": context
        }

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("answer", answer_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", END)

    return workflow.compile()