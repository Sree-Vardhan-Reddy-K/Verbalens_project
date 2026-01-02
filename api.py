from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from main import prepare_knowledge_base, get_rag_response

# App setup
load_dotenv()

app = FastAPI(title="Verbalens API", version="1.0")

# Startup: build retriever once
DATA_DIR =  "data"

if not os.path.exists(DATA_DIR):
    raise RuntimeError("data/ directory not found")

retriever = prepare_knowledge_base(DATA_DIR)

# Request / Response schema
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

# Health check (important for EC2)
@app.get("/health")
def health():
    return {"status": "ok"}

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
def query_docs(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer, sources = get_rag_response(
            prompt=payload.question,
            retriever=retriever,
            eval_mode=False
        )

        return {
            "answer": answer,
            "sources": sources
        }


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
