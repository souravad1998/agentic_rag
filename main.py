"""Vedic AI Life Coach — CLI entry point."""

from src import EmbeddingManager, VectorStore, RAGRetriever, AdvancedRAGPipeline


def main():
    print("=" * 60)
    print("🙏  Initializing Vedic AI Life Coach")
    print("=" * 60)

    embedding_manager = EmbeddingManager()
    vectorstore = VectorStore()
    rag_retriever = RAGRetriever(vectorstore, embedding_manager)
    coach = AdvancedRAGPipeline(rag_retriever)

    print("\n" + "=" * 60)
    print("🙏  Welcome to the Vedic AI Life Coach")
    print("    Ask any question about life, and I'll guide you")
    print("    with wisdom from the Bhagavad Gita.")
    print("    Type 'quit' to exit.")
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
