"""CSV loader + text splitter for ingesting scripture data."""

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CSV_FILE_PATH


def load_documents(file_path: str = CSV_FILE_PATH):
    """Load verses from CSV. Returns LangChain Document objects."""
    loader = CSVLoader(
        file_path=file_path,
        source_column="chapter_verse",
        metadata_columns=["chapter_title"],
        content_columns=["chapter_number", "chapter_verse", "translation"],
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split docs into chunks. Most Gita verses fit in one chunk anyway."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} docs → {len(chunks)} chunks")
    return chunks
