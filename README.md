# Climate Methodology Assistant (RAG System)

Этот проект — простой Retrieval-Augmented Generation (RAG) пайплайн для поиска информации по климатическим методологиям (VM-стандарты и связанные документы).

Система отвечает на вопросы, опираясь только на загруженные документы, без внешних знаний.

---

## Что делает система

1. Делит документы на фрагменты (chunking)
2. Строит эмбеддинги для каждого фрагмента
3. Индексирует их в FAISS
4. По запросу пользователя:
   - ищет похожие фрагменты (retrieval)
   - собирает контекст
   - передаёт его в FLAN-T5
   - генерирует ответ только по контексту

---

## Архитектура

```

data/processed → chunking → embeddings → FAISS index
↓
semantic search
↓
retrieved context
↓
FLAN-T5 generator
↓
answer

````

---

## Основные компоненты

### 1. Chunking
`src/ingest/optimal_chunking.py`

- Делит тексты на токенизированные чанки
- Использует sliding window
- Добавляет metadata (source, chunk_id)

---

### 2. Индексация
`src/ingest/build_index.py`

- Загружает тексты
- Создаёт эмбеддинги (BERT / sentence-transformers)
- Строит FAISS index
- Сохраняет:
  - `index.faiss`
  - `chunks.npy`
  - `metadata.npy`

---

### 3. Retrieval
`src/retrieval/semantic_search.py`

- Преобразует запрос в embedding
- Ищет ближайшие векторы через FAISS
- Возвращает top-k релевантных фрагментов

---

### 4. Генерация ответа
`src/llm/flan_generator.py`

- Используется `google/flan-t5-base`
- Получает:
  - вопрос
  - контекст из retrieval
- Генерирует ответ строго по контексту

---

### 5. Тестовый запуск RAG
`test_rag.py`

- запускает retrieval
- собирает контекст
- вызывает LLM
- выводит ответ и источники

---

## Как запустить

### 1. Установить зависимости
```bash
pip install -r requirements.txt
````

### 2. Подготовить данные

Поместить `.txt` файлы в:

```
data/processed/
```

### 3. Построить индекс

```bash
python -m src.ingest.build_index
```

### 4. Запустить тест RAG

```bash
python test_rag.py
```

---

## Используемые технологии

* FAISS — поиск ближайших векторов
* sentence-transformers — эмбеддинги
* HuggingFace Transformers — FLAN-T5 модель
* PyTorch — инференс модели
* NumPy — хранение и обработка данных

---

## Примечания

* Индекс FAISS и данные не хранятся в Git (генерируются локально)
* При первом запуске модели скачиваются с HuggingFace Hub
* Возможны предупреждения от transformers — они не критичны

---

## Ограничения

* Нет GPU-оптимизации (все работает на CPU по умолчанию)
* Простая схема chunking (без семантического разбиения)
* FAISS IndexFlatIP (без HNSW или IVF)

