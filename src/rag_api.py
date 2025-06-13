from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag.vector_store import VectorStore
from rag.rag_pipeline import RagPipeline

app = FastAPI()

# Load the pipeline ONCE at startup
vector_store = VectorStore()
rag = RagPipeline(vector_store=vector_store)

class QueryRequest(BaseModel):
    query: str

@app.post("/answer")
async def answer(request: QueryRequest):
    result = rag.answer(request.query)
    return {"answer": result}
