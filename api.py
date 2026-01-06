from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from fastapi import UploadFile, File
import shutil

_retriever = None
_is_index_ready = False

def lazy_prepare_knowledge_base(*args, **kwargs):
    from main import prepare_knowledge_base
    return prepare_knowledge_base(*args, **kwargs)

def lazy_get_rag_response(*args, **kwargs):
    from main import get_rag_response
    return get_rag_response(*args, **kwargs)

# App setup
load_dotenv()

app = FastAPI(title="Verbalens API", version="1.0")

# Startup: build retriever once
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)


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

_retriever = None

def get_retriever(file_paths):
    global _retriever, _is_index_ready
    if _retriever is None:
        _retriever = lazy_prepare_knowledge_base(file_paths)
        _is_index_ready = True
    return _retriever


# Main query endpoint
@app.post("/query", response_model=QueryResponse)
def query_docs(payload: QueryRequest):
    if not _is_index_ready:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload PDFs first."
        )
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        retriever = get_retriever([DATA_DIR])
        answer, sources =lazy_get_rag_response(
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

    DEFAULT_PDF_PATH = os.path.join(DATA_DIR, "company_policy.pdf")

    if not files:
        if not os.path.exists(DEFAULT_PDF_PATH):
            raise HTTPException(
                status_code=400,
                detail="Default PDF not found on server."
            )
        saved_paths = [DEFAULT_PDF_PATH]
    else:
        saved_paths = []
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Only PDFs allowed")

            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            saved_paths.append(file_path)

    # rebuild retriever from uploaded PDFs
    retriever =  get_retriever(saved_paths)

    return {
        "status": "success",
        "files_indexed": len(saved_paths)
    }
