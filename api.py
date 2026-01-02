from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from fastapi import UploadFile, File
import shutil


from main import prepare_knowledge_base, get_rag_response

# App setup
load_dotenv()

app = FastAPI(title="Verbalens API", version="1.0")

# Startup: build retriever once
DATA_DIR =  "data"

if not os.path.exists(DATA_DIR):
    raise RuntimeError("data/ directory not found")

retriever = prepare_knowledge_base("data")

# Request / Response schema
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

# Health check
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


UPLOAD_DIR = "temp_pdf_store"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
def upload_pdfs(files: list[UploadFile] = File(...)):
    global retriever

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    saved_paths = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDFs allowed")

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        saved_paths.append(file_path)

    # rebuild retriever from uploaded PDFs
    retriever = prepare_knowledge_base(saved_paths)

    return {
        "status": "success",
        "files_indexed": len(saved_paths)
    }
