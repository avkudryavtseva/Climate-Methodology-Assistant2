from src.retrieval.semantic_search import SemanticSearchSystem


text = """
Biochar methodologies require permanence monitoring.

Soil carbon projects need baseline calculations.

Additionality is required for carbon credits.

Leakage assessment is important in forestry projects.
"""

system = SemanticSearchSystem()

system.index_document(text)

results = system.search(
    "How is permanence monitored?"
)

print(results)

