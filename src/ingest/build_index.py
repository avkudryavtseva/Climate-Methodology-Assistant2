import os
import numpy as np
import faiss

from src.ingest.optimal_chunking import build_corpus
from src.embeddings.bert_embeddings import BertEmbedder
# from src.embeddings.vector_store import VectorStore


# папка с обработанными txt-файлами
DATA_PATH = "data/processed"

# папка для сохранения FAISS index
INDEX_PATH = "data/faiss_index"


def build_index():

    # создаём chunks и metadata из документов
    print("Building corpus...")

    chunks, metadata = build_corpus(DATA_PATH)

    print(f"Total chunks: {len(chunks)}")

    # загружаем embedding-модель
    print("Loading embedder...")

    embedder = BertEmbedder()

    # создаём embeddings для всех чанков
    print("Creating embeddings...")

    embeddings = embedder.encode(chunks)

    # переводим embeddings в float32 для FAISS
    embeddings = np.array(embeddings).astype("float32")

    # создаём FAISS index для semantic search
    print("Building FAISS index...")

    # размер embedding vector
    dimension = embeddings.shape[1]

    # создаём index для similarity search
    index = faiss.IndexFlatIP(dimension)

    # нормализуем embeddings для cosine similarity
    faiss.normalize_L2(embeddings)

    # добавляем embeddings в index
    index.add(embeddings)

    # сохраняем index и metadata
    print("Saving index...")

    os.makedirs(INDEX_PATH, exist_ok=True)

    # сохраняем FAISS index
    faiss.write_index(index, f"{INDEX_PATH}/index.faiss")

    # сохраняем тексты чанков
    np.save(
        f"{INDEX_PATH}/chunks.npy",
        np.array(chunks, dtype=object)
    )

    # сохраняем metadata
    np.save(
        f"{INDEX_PATH}/metadata.npy",
        np.array(metadata, dtype=object)
    )

    print("DONE")


if __name__ == "__main__":

    # запускаем полную сборку index
    build_index()
