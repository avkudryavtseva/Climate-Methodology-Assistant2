import os
from typing import List, Dict
from transformers import AutoTokenizer


# настройки chunking
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# максимальный размер чанка
MAX_CHUNK_TOKENS = 300

# overlap между чанками для сохранения контекста
OVERLAP = 50


# загружаем tokenizer модели
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# считаем количество токенов в тексте
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# переводим текст в токены
def encode(text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


# переводим токены обратно в текст
def decode(tokens: List[int]) -> str:
    return tokenizer.decode(tokens)


# очищаем текст от лишних пробелов и переносов
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


# разбиваем текст на предложения
def split_sentences(text: str) -> List[str]:

    import re

    sentences = re.split(r'(?<=[.!?])\s+', text)

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 0
    ]


# режем длинный текст на чанки фиксированного размера
def chunk_tokens(
    tokens: List[int],
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap: int = OVERLAP
) -> List[List[int]]:

    chunks = []

    i = 0

    while i < len(tokens):

        # берём окно токенов
        chunk = tokens[i:i + max_tokens]

        chunks.append(chunk)

        # двигаем окно с overlap
        i += max_tokens - overlap

    return chunks


# обрабатываем один txt-файл
def build_chunks_from_file(file_path: str) -> List[Dict]:

    # читаем файл
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # очищаем текст
    text = clean_text(text)

    # переводим текст в токены
    tokens = encode(text)

    # создаём чанки
    token_chunks = chunk_tokens(tokens)

    final_chunks = []

    # проходим по всем чанкам
    for idx, chunk_tokens_list in enumerate(token_chunks):

        # переводим токены обратно в текст
        chunk_text = decode(chunk_tokens_list).strip()

        # пропускаем слишком короткие чанки
        if len(chunk_text) < 50:
            continue

        # сохраняем текст и metadata
        final_chunks.append({

            "text": chunk_text,

            "metadata": {

                # сохраняем источник чанка
                "source": os.path.basename(file_path),

                # сохраняем номер чанка
                "chunk_id": idx,

                # сохраняем размер чанка
                "token_len": len(chunk_tokens_list)
            }
        })

    return final_chunks


# собираем общий корпус документов
def build_corpus(folder_path: str):

    all_chunks = []
    metadata = []

    # проходим по всем txt-файлам
    for file in os.listdir(folder_path):

        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(folder_path, file)

        # создаём чанки для файла
        chunks = build_chunks_from_file(file_path)

        # сохраняем чанки и metadata
        for c in chunks:

            all_chunks.append(c["text"])
            metadata.append(c["metadata"])

    return all_chunks, metadata
