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
    needs_retry: bool
    retry_count: int


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
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("answer", answer_node)

    workflow.add_edge(START, "retrieve")

    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "rewrite": "rewrite",
            "answer": "answer"
        }
    )

    workflow.add_edge("rewrite", "retrieve")

    workflow.add_edge("answer", END)

    return workflow.compile()

def grade_node(state):
    context = state["context"]
    needs_retry = len(context) < 1000
    return {
        "needs_retry": needs_retry
    }

def route_after_grade(state):
    if state.get("retry_count", 0) >= 1:
        return "answer"
    if state["needs_retry"]:
        return "rewrite"
    return "answer"

def rewrite_node(state):

    return {
        "question": (
            f"Provide detailed information about "
            f"{state['question']}"
        ),
        "retry_count": state.get(
            "retry_count",
            0
        ) + 1
    }