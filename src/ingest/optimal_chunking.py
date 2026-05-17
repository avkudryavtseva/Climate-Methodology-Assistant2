import os
from typing import List, Dict
from transformers import AutoTokenizer

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_CHUNK_TOKENS = 300
OVERLAP = 50

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# -----------------------------
# TOKEN UTILS
# -----------------------------
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def encode(text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def decode(tokens: List[int]) -> str:
    return tokenizer.decode(tokens)


# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


# -----------------------------
# SPLIT INTO SENTENCES
# -----------------------------
def split_sentences(text: str) -> List[str]:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]


# -----------------------------
# TOKEN-BASED CHUNKING (CORE)
# -----------------------------
def chunk_tokens(tokens: List[int],
                 max_tokens: int = MAX_CHUNK_TOKENS,
                 overlap: int = OVERLAP) -> List[List[int]]:

    chunks = []
    i = 0

    while i < len(tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(chunk)

        i += max_tokens - overlap

    return chunks


# -----------------------------
# MAIN FILE PROCESSING
# -----------------------------
def build_chunks_from_file(file_path: str) -> List[Dict]:

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_text(text)

    tokens = encode(text)

    token_chunks = chunk_tokens(tokens)

    final_chunks = []

    for idx, chunk_tokens_list in enumerate(token_chunks):

        chunk_text = decode(chunk_tokens_list).strip()

        # skip garbage
        if len(chunk_text) < 50:
            continue

        final_chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": os.path.basename(file_path),
                "chunk_id": idx,
                "token_len": len(chunk_tokens_list)
            }
        })

    return final_chunks


# -----------------------------
# CORPUS BUILDER
# -----------------------------
def build_corpus(folder_path: str):

    all_chunks = []
    metadata = []

    for file in os.listdir(folder_path):

        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(folder_path, file)

        chunks = build_chunks_from_file(file_path)

        for c in chunks:
            all_chunks.append(c["text"])
            metadata.append(c["metadata"])

    return all_chunks, metadata
