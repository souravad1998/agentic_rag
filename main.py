"""
Vedic AI Life Coach — CLI Entry Point

Wires all modules together and provides an interactive chat interface.
"""

from src import EmbeddingManager, VectorStore, RAGRetriever, AdvancedRAGPipeline


def main():
    """Initialize all components and start the interactive chat."""

    # Step 1: Initialize components
    print("=" * 60)
    print("🙏  Initializing Vedic AI Life Coach")
    print("=" * 60)

    embedding_manager = EmbeddingManager()
    vectorstore = VectorStore()
    rag_retriever = RAGRetriever(vectorstore, embedding_manager)
    coach = AdvancedRAGPipeline(rag_retriever)

    # Step 2: Interactive chat loop
    print("\n" + "=" * 60)
    print("🙏  Welcome to the Vedic AI Life Coach\n")
    print("    Ask any question about life, and I'll guide you\n")
    print("    with wisdom from the Bhagavad Gita.\n")
    print("    Type 'quit' to exit.\n")
    print("=" * 60 + "\n")

    while True:
        question = input("🧑 You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\n🙏 May the wisdom of the Gita guide your path. Namaste!")
            break

        result = coach.query(question)
        coach.display(result)
        print()


if __name__ == "__main__":
    main()
