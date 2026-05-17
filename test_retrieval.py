from src.retrieval.semantic_search import SemanticSearchSystem

search = SemanticSearchSystem()

query = "How is permanence ensured in forestry projects?"

results = search.search(query, top_k=5)

for r in results:
    print("\n---")
    print(r["source"])
    print(r["score"])
    print(r["text"][:300])
