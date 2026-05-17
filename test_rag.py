import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from src.retrieval.semantic_search import SemanticSearchSystem
from src.llm.flan_generator import FlanT5Generator


def format_context(results):
    context = ""
    sources = set()

    for r in results:
        context += r["text"] + "\n\n"
        sources.add(r["source"])

    return context, sources


def main():

    search = SemanticSearchSystem()
    llm = FlanT5Generator()

    question = "How to calculate the baseline?"

    print("\n================ QUESTION ================\n")
    print(question)

    # 1. RETRIEVAL
    results = search.search(question, top_k=5)

    print("\nTOP RESULTS:")
    for r in results:
        print(r["score"], r["source"])

    # 2. CONTEXT
    context, sources = format_context(results)

    # 3. LLM CALL (ВАЖНО: 2 аргумента!)
    answer = llm.generate(question, context)

    # 4. OUTPUT
    print("\n================ ANSWER ================\n")
    print(answer)

    print("\n================ SOURCES ================\n")
    for s in sources:
        print(s)


if __name__ == "__main__":
    main()
