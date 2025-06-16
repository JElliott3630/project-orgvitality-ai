from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag.vector_store import VectorStore
from rag.rag_pipeline import RagPipeline

app = FastAPI()

# --- Your Hardcoded API Key ---
API_KEY = "f4a7b9d8-1e2c-4f5a-8a6b-3c7d9e1f2a3b"

# --- Middleware to handle Cross-Origin Resource Sharing (CORS) ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pipeline ONCE at startup
vector_store = VectorStore()
rag = RagPipeline(vector_store=vector_store)

class QueryRequest(BaseModel):
    query: str

@app.post("/answer")
async def answer(
    request: QueryRequest,
    x_api_key: str | None = Header(None, alias="X-API-Key")
):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    result = rag.answer(request.query)
    return {"answer": result}

@app.post("/answer-stream")
async def answer_stream(
    request: QueryRequest,
    x_api_key: str | None = Header(None, alias="X-API-Key")
):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    return StreamingResponse(rag.answer_stream(request.query), media_type="text/plain")