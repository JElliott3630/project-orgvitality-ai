import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Header, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Local Imports ---
from src.rag.rag_pipeline import RagPipeline
from src.rag.vector_store import PineconeVectorStore  # Adjust if location differs
from src.config import SUPABASE_URL
from src.auth import get_current_user

# Load environment variables
load_dotenv()

# --- Globals ---
rag_pipeline_instance: RagPipeline | None = None

# --- Startup/Shutdown Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline_instance
    print("Application startup: Initializing RAG Pipeline...")
    vector_store = PineconeVectorStore(user_id="orgvitality-default")
    rag_pipeline_instance = RagPipeline(vector_store=vector_store, use_reranker=False)
    print("[INFO] Asynchronous RagPipeline initialized (reranking is DISABLED).")
    yield
    print("Application shutdown.")

app = FastAPI(lifespan=lifespan)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

# --- Protected API Router (requires valid JWT token) ---
protected_router = APIRouter(
    prefix="/orgvitality",
    dependencies=[Depends(get_current_user)],
)

@protected_router.post("/answer")
async def answer(request: QueryRequest):
    if not rag_pipeline_instance:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized.")
    result = await rag_pipeline_instance.answer(request.query)
    return {"answer": result}

@protected_router.post("/answer-stream")
async def answer_stream(request: QueryRequest):
    if not rag_pipeline_instance:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized.")
    stream = rag_pipeline_instance.answer_stream(request.query)
    return StreamingResponse(stream, media_type="text/plain")

@protected_router.post("/test-auth")
async def test_auth(request: Request):
    user = await get_current_user(request)
    return {"user": user}

# --- Public Test Endpoint (uses API key) ---
@app.post("/test-answer")
async def test_answer(
    request: QueryRequest,
    x_api_key: str = Header(..., alias="X-API-Key")
):
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="API key not set in environment.")
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    if not rag_pipeline_instance:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized.")
    result = await rag_pipeline_instance.answer(request.query)
    return {"answer": result}

# --- Mount the protected router ---
app.include_router(protected_router)
