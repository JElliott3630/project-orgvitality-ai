import asyncio
import httpx
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Local Imports ---
from src.rag.vector_store import VectorStore
from src.rag.rag_pipeline import RagPipeline
from src.config import SUPABASE_URL, SUPABASE_ANON_KEY

# Load environment variables from .env file
load_dotenv()

# --- Globals for the application ---
rag_pipeline_instance: RagPipeline | None = None
http_bearer_scheme = HTTPBearer()

# --- Conditional Authentication ---
# Check an environment variable to see if auth should be enforced.
# This makes local testing much easier without sacrificing production security.
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"

async def get_current_user_real(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer_scheme),
) -> dict:
    """
    Verifies the Supabase JWT token and returns the user data.
    This is the REAL authentication check for production.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(
            status_code=500,
            detail="Supabase URL or Anon Key is not configured on the server.",
        )

    token = credentials.credentials
    user_url = f"{SUPABASE_URL}/auth/v1/user"
    headers = {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(user_url, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail="Invalid authentication credentials.",
            )
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Service unavailable.")

    return response.json()

async def get_current_user_mock() -> dict:
    """A mock user for easy local testing when authentication is disabled."""
    print("--- AUTHENTICATION DISABLED ---")
    return {"email": "local-test-user@example.com", "id": "mock-user-id"}

# Use a different dependency based on the environment variable
if AUTH_ENABLED:
    get_current_user = get_current_user_real
    print("Authentication is ENABLED.")
else:
    get_current_user = get_current_user_mock
    print("Authentication is DISABLED for local testing.")


# --- Application Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup events. Initializes the RAG pipeline.
    """
    global rag_pipeline_instance
    print("Application startup: Initializing RAG Pipeline...")
    vector_store = VectorStore()
    rag_pipeline_instance = RagPipeline(vector_store=vector_store, use_reranker=False)
    asyncio.create_task(rag_pipeline_instance.initialize_vector_store())
    yield
    print("Application shutdown.")


app = FastAPI(lifespan=lifespan)

# --- Middleware for CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

# --- API Endpoints ---
@app.post("/answer")
async def answer(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user),
):
    """Handles a query and returns a single, complete answer."""
    if not rag_pipeline_instance:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")
    
    print(f"Request received from user: {current_user.get('email')}")
    result = await rag_pipeline_instance.answer(request.query)
    return {"answer": result}


@app.post("/answer-stream")
async def answer_stream(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user),
):
    """Handles a query and returns the answer as a stream."""
    if not rag_pipeline_instance:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")
    
    print(f"Streaming request received from user: {current_user.get('email')}")
    stream = rag_pipeline_instance.answer_stream(request.query)
    return StreamingResponse(stream, media_type="text/plain")

