"""
Data Loader — loads Bhagavad Gita CSV and splits documents into chunks.
"""

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CSV_FILE_PATH


def load_documents(file_path: str = CSV_FILE_PATH):
    """
    Load Bhagavad Gita verses from CSV using LangChain CSVLoader.

    Returns:
        List of LangChain Document objects
    """
    loader = CSVLoader(
        file_path=file_path,
        source_column="chapter_verse",
        metadata_columns=["chapter_title"],
        content_columns=["chapter_number", "chapter_verse", "translation"],
    )
    documents = loader.load()
    print(f"Total documents loaded: {len(documents)}")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: List of LangChain Documents
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked Documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print(f"\nExample Chunk:")
        print(split_docs[0].page_content)
        print("-" * 50)
        print(split_docs[0].metadata)

    return split_docs
