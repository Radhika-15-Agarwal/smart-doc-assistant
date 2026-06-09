import os
import streamlit as st

from rag import (
    load_pdfs,
    split_documents,
    create_vector_store,
    load_vector_store
)

from graph import build_graph


st.set_page_config(
    page_title="Smart PDF Assistant",
    page_icon="📄"
)

st.title("📄 Smart PDF Assistant")
st.caption(
    "Ask questions about PDF documents using LangGraph and RAG."
)


# -------------------------
# Upload PDF
# -------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    os.makedirs("data", exist_ok=True)

    file_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(
            uploaded_file.getbuffer()
        )

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


# -------------------------
# Process PDF
# -------------------------

if st.button("Process PDFs"):

    with st.spinner("Processing PDFs..."):

        docs = load_pdfs()

        chunks = split_documents(docs)

        create_vector_store(chunks)

    st.success(
        "Vector database created!"
    )


# -------------------------
# Cache Graph
# -------------------------

@st.cache_resource
def get_graph():

    vector_store = load_vector_store()

    return build_graph(
        vector_store
    )


# -------------------------
# Ask Question
# -------------------------

question = st.text_input(
    "Ask a question"
)

if st.button("Ask"):

    if question:

        graph = get_graph()

        with st.spinner("Thinking..."):

            result = graph.invoke(
                {
                    "question": question,
                    "retry_count": 0
                }
            )

        st.divider()

        st.subheader("Answer")

        st.write(
            result["answer"]
        )