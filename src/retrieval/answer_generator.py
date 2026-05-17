def generate_answer(query, results):

    top_chunks = results[:3]

    context = "\n\n".join([
        r["text"] for r in top_chunks
    ])

    answer = f"""
QUESTION:
{query}

ANSWER (based on Verra methodologies):
{context}
"""

    return answer
