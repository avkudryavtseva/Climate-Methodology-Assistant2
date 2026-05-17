from sentence_transformers import SentenceTransformer


class BertEmbedder:

    def __init__(self):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def encode(self, texts):

        embeddings = self.model.encode(texts)

        return embeddings
