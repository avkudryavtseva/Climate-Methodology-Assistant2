import os

# отключаем лишние warning'и от tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from src.retrieval.semantic_search import SemanticSearchSystem
from src.llm.flan_generator import FlanT5Generator


def format_context(results):
    """
    - cобираем контекст из retrieved chunks.
    - склеиваем тексты чанков в один prompt
    - собираем уникальные источники
    """
    context = ""
    sources = set()

    for r in results:

        # добавляем текст найденного chunk'а в общий контекст
        context += r["text"] + "\n\n"

        # сохраняем источник документа
        sources.add(r["source"])

    return context, sources


def main():

    # инициализируем retrieval систему (FAISS + embeddings)
    search = SemanticSearchSystem()

    # инициализируем LLM (FLAN-T5)
    llm = FlanT5Generator()

    # тестовый вопрос
    question = "How to calculate the baseline?"

    print("\n================ QUESTION ================\n")
    print(question)

    # 1. RETRIEVAL STEP
    # ищем наиболее похожие чанки в FAISS
    results = search.search(question, top_k=5)

    # выводим top результатов для отладки
    print("\nTOP RESULTS:")
    for r in results:
        print(r["score"], r["source"])

    # 2. CONTEXT BUILDING
    # превращаем retrieved chunks в единый текстовый контекст
    context, sources = format_context(results)

    # 3. GENERATION STEP (LLM)
    # передаём вопрос + контекст в FLAN-T5
    answer = llm.generate(question, context)

    # 4. OUTPUT
    print("\n================ ANSWER ================\n")
    print(answer)

    print("\n================ SOURCES ================\n")
    for s in sources:
        print(s)


if __name__ == "__main__":
    main()
