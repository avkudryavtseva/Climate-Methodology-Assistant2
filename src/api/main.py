print("IMPORT MAIN OK")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.retrieval.semantic_search import SemanticSearchSystem
from src.llm.flan_generator import FlanT5Generator

app = FastAPI()


# -------------------------
# STATIC FRONTEND
# -------------------------

app.mount(
    "/static",
    StaticFiles(directory="src/api/static"),
    name="static"
)


@app.get("/")
def root():
    return FileResponse("src/api/static/index.html")


# -------------------------
# MODELS
# -------------------------

search = None
llm = None


@app.on_event("startup")
def startup():
    global search, llm
    search = SemanticSearchSystem()
    llm = FlanT5Generator()


# -------------------------
# API SCHEMA
# -------------------------

class QuestionRequest(BaseModel):
    question: str


# -------------------------
# RAG ENDPOINT
# -------------------------

@app.post("/ask")
def ask(req: QuestionRequest):
    results = search.search(req.question, top_k=5)

    context = "\n\n".join([r["text"] for r in results])

    answer = llm.generate(req.question, context)

    return {
        "answer": answer,
        "sources": [r["source"] for r in results]
    }
