from src.retrieval.semantic_search import SemanticSearchSystem
from src.retrieval.answer_generator import generate_answer


system = SemanticSearchSystem()

system.index_corpus("data/processed")

query = "How is permanence ensured in forestry carbon projects?"

results = system.search(query)

print("\n================ CLIMATE ASSISTANT =================\n")

print(generate_answer(query, results))

print("\n================ SOURCES =================\n")

for r in results:

    print("SOURCE:", r["metadata"]["source"])
    print("SCORE:", round(r["score"], 3))
    print("-" * 40)
