import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class SemanticSearchSystem:

    def __init__(self):
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # FAISS index
        self.vector_store = faiss.read_index("data/faiss_index/index.faiss")

        # chunks + metadata
        self.chunks = np.load(
            "data/faiss_index/chunks.npy",
            allow_pickle=True
        )
        self.metadata = np.load(
            "data/faiss_index/metadata.npy",
            allow_pickle=True
        )

    def search(self, query, top_k=5):

        # 1. embedding
        query_embedding = self.embedder.encode(
            query,
            normalize_embeddings=True
        )

        query_embedding = np.array(
            query_embedding,
            dtype="float32"
        ).reshape(1, -1)

        # 2. FAISS search
        scores, indices = self.vector_store.search(
            query_embedding,
            top_k
        )

        # 3. results
        results = []

        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue

            results.append({
                "text": self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "score": float(scores[0][i])
            })

        # 4. sort (FAISS иногда уже сортирует, но лучше явно)
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results
