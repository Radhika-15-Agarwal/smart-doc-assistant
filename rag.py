from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

def load_pdfs(folder_path="data"):
    documents = []

    pdf_files = list(Path(folder_path).glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDFs")

    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}")

        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)

def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vector_store

def load_vector_store():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )


def answer_question(question, vector_store):

    docs = retrieve_documents(
        question,
        vector_store
    )

    context = build_context(docs)

    answer = generate_answer(
        question,
        context
    )

    sources = sorted(
        set(
            f"Page {doc.metadata.get('page', 0) + 1}"
            for doc in docs
        )
    )

    return answer, sources

def retrieve_documents(question, vector_store):

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 20
        }
    )

    return retriever.invoke(question)

def build_context(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

def generate_answer(question, context):

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash"
        )

        prompt = f"""
        Answer ONLY using the provided context.

        Context:
        {context}

        Question:
        {question}
        """

        response = llm.invoke(prompt)

        return response.content

    except Exception:
        return context[:1000]