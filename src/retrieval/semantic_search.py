import numpy as np
import faiss

# модель для создания embeddings
from sentence_transformers import SentenceTransformer


class SemanticSearchSystem:

    def __init__(self):

        # загружаем embedding-модель
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # загружаем FAISS index для semantic search
        self.vector_store = faiss.read_index(
            "data/faiss_index/index.faiss"
        )

        # загружаем тексты чанков
        self.chunks = np.load(
            "data/faiss_index/chunks.npy",
            allow_pickle=True
        )

        # загружаем metadata чанков
        self.metadata = np.load(
            "data/faiss_index/metadata.npy",
            allow_pickle=True
        )

    def search(self, query, top_k=5):

        # создаём embedding для user query
        query_embedding = self.embedder.encode(
            query,

            # нормализуем embeddings для cosine similarity
            normalize_embeddings=True
        )

        # переводим embedding в формат для FAISS
        query_embedding = np.array(
            query_embedding,
            dtype="float32"
        ).reshape(1, -1)

        # ищем наиболее похожие чанки
        scores, indices = self.vector_store.search(
            query_embedding,
            top_k
        )

        # собираем результаты поиска
        results = []

        for i, idx in enumerate(indices[0]):

            # пропускаем пустые результаты
            if idx < 0:
                continue

            results.append({

                # текст найденного чанка
                "text": self.chunks[idx],

                # источник чанка
                "source": self.metadata[idx]["source"],

                # similarity score
                "score": float(scores[0][i])
            })

        # сортируем результаты по similarity score
        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )

        return results
