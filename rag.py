from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(folder_path="data"):
    documents = []

    pdf_files = list(Path(folder_path).glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDFs")

    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}")

        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    return documents