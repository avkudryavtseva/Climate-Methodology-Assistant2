import os
import numpy as np
import faiss

from src.ingest.optimal_chunking import build_corpus
from src.embeddings.bert_embeddings import BertEmbedder
from src.embeddings.vector_store import VectorStore


DATA_PATH = "data/processed"
INDEX_PATH = "data/faiss_index"


def build_index():

    print("📦 Building corpus...")

    chunks, metadata = build_corpus(DATA_PATH)

    print(f"Total chunks: {len(chunks)}")

    print("🧠 Loading embedder...")

    embedder = BertEmbedder()

    print("🔢 Creating embeddings...")

    embeddings = embedder.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    print("📚 Building FAISS index...")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print("💾 Saving index...")

    os.makedirs(INDEX_PATH, exist_ok=True)

    faiss.write_index(index, f"{INDEX_PATH}/index.faiss")

    np.save(f"{INDEX_PATH}/chunks.npy", np.array(chunks, dtype=object))
    np.save(f"{INDEX_PATH}/metadata.npy", np.array(metadata, dtype=object))

    print("✅ DONE")


if __name__ == "__main__":
    build_index()

