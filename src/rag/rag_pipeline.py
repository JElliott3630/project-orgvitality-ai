import openai
import json
import yaml
import logging
import asyncio
from sentence_transformers import CrossEncoder
from src.config import OPENAI_API_KEY, PROMPT_PATH # Using absolute import

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

def load_prompts(path=PROMPT_PATH):
    """Loads prompts from a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

class RagPipeline:
    """
    An asynchronous RAG (Retrieval-Augmented Generation) pipeline
    with lazy loading and a toggle for the reranker.
    """
    def __init__(self, vector_store, prompts=None, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2", use_reranker: bool = False):
        self.vector_store = vector_store
        self.prompts = prompts or load_prompts()
        self.async_client = openai.AsyncClient(api_key=OPENAI_API_KEY)
        
        # --- MODIFIED ---
        # Added a flag to control reranking and still support lazy loading.
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model
        self.cross_encoder = None 
        # --- END MODIFIED ---

        if self.use_reranker:
            logging.info("Asynchronous RagPipeline initialized (reranker will be loaded on first use).")
        else:
            logging.info("Asynchronous RagPipeline initialized (reranking is DISABLED).")


    async def _load_reranker(self):
        """Loads the CrossEncoder model on demand."""
        # Only load if reranking is enabled and the model hasn't been loaded yet.
        if self.use_reranker and self.cross_encoder is None:
            logging.info(f"First use: Lazily loading reranker model '{self.reranker_model_name}'...")
            # Run the synchronous, slow model loading in a separate thread
            self.cross_encoder = await asyncio.to_thread(CrossEncoder, self.reranker_model_name)
            logging.info("Reranker model loaded.")

    async def initialize_vector_store(self, rebuild=False):
        """Asynchronously checks and builds the vector store."""
        is_built = await asyncio.to_thread(self.vector_store.is_built)
        if rebuild or not is_built:
            logging.info("Building or rebuilding the vector store...")
            await asyncio.to_thread(self.vector_store.build_from_json)
        else:
            logging.info("Vector store already built, skipping rebuild.")

    async def expand_query(self, user_query: str) -> list[str]:
        # This method remains unchanged
        sys_prompt = self.prompts["query_expansion"]["system_prompt"]
        user_prompt = self.prompts["query_expansion"]["user_prompt_template"].format(query_text=user_query)
        try:
            response = await self.async_client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}])
            expanded = response.choices[0].message.content.strip()
            subqueries = json.loads(expanded)
            return subqueries
        except (json.JSONDecodeError, openai.APIError) as e:
            logging.warning("Query expansion failed, using original query. Error: %s", str(e))
            return [user_query]

    async def retrieve(self, queries: list[str], k: int = 8) -> list[dict]:
        # This method remains unchanged
        retrieved_chunks = []
        for q in queries:
            results = await asyncio.to_thread(self.vector_store.query, q, n_results=k)
            docs, metadatas = results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]
            for doc, meta in zip(docs, metadatas):
                retrieved_chunks.append({"text": doc, "metadata": meta, "query": q})
        return retrieved_chunks

    async def rerank(self, user_query: str, retrieved_chunks: list[dict], top_n: int = 6) -> list[dict]:
        """Asynchronously reranks retrieved chunks, loading the model on first use."""
        if not retrieved_chunks:
            return []
            
        await self._load_reranker()

        pairs = [(user_query, chunk["text"]) for chunk in retrieved_chunks]
        scores = await asyncio.to_thread(self.cross_encoder.predict, pairs, show_progress_bar=False)
        
        scored_chunks = list(zip(retrieved_chunks, scores))
        reranked = sorted(scored_chunks, key=lambda x: x[1], reverse=True)[:top_n]
        return [chunk for chunk, score in reranked]

    async def generate_answer(self, user_query: str, context_chunks: list[dict]):
        sys_prompt = self.prompts["answer_generation"]["system_prompt"]
        def format_chunk_for_context(chunk: dict) -> str:
            label_parts = [f"Source: {chunk['metadata']['source']}"] if "source" in chunk.get('metadata', {}) else []
            if chunk.get('metadata', {}).get("source_detail"): label_parts.append(str(chunk['metadata']["source_detail"]))
            return f"[{', '.join(label_parts)}]\n{chunk['text']}"
        context_block = "\n\n".join([format_chunk_for_context(chunk) for chunk in context_chunks])
        user_prompt = self.prompts["answer_generation"]["user_prompt_template"].format(query_text=user_query, context=context_block)
        response = await self.async_client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}], stream=False)
        return response.choices[0].message.content.strip()

    async def generate_answer_stream(self, user_query: str, context_chunks: list[dict]):
        sys_prompt = self.prompts["answer_generation"]["system_prompt"]
        def format_chunk_for_context(chunk: dict) -> str:
            label_parts = [f"Source: {chunk['metadata']['source']}"] if "source" in chunk.get('metadata', {}) else []
            if chunk.get('metadata', {}).get("source_detail"): label_parts.append(str(chunk['metadata']["source_detail"]))
            return f"[{', '.join(label_parts)}]\n{chunk['text']}"
        context_block = "\n\n".join([format_chunk_for_context(chunk) for chunk in context_chunks])
        user_prompt = self.prompts["answer_generation"]["user_prompt_template"].format(query_text=user_query, context=context_block)
        stream = await self.async_client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}], stream=True)
        async for chunk in stream:
            if content := chunk.choices[0].delta.content:
                yield content

    async def _get_full_pipeline_response(self, user_query: str) -> list[dict]:
        """Helper to run the retrieval and optional reranking pipeline."""
        # This value was the default in the rerank method.
        TOP_N_FINAL_CHUNKS = 6

        subqueries = await self.expand_query(user_query)
        retrieved = await self.retrieve(subqueries)
        # Deduplicate the chunks based on their text content
        unique_chunks = list({chunk["text"]: chunk for chunk in retrieved}.values())

        # --- MODIFIED ---
        # Conditionally execute the reranking step
        if self.use_reranker:
            logging.info("Reranking retrieved chunks...")
            final_chunks = await self.rerank(user_query, unique_chunks, top_n=TOP_N_FINAL_CHUNKS)
        else:
            logging.info("Skipping reranking. Taking top chunks from retrieval.")
            # If not reranking, just take the first N chunks after deduplication.
            final_chunks = unique_chunks[:TOP_N_FINAL_CHUNKS]
        # --- END MODIFIED ---
        
        return final_chunks

    async def answer(self, user_query: str) -> str:
        """Runs the full pipeline and returns a single answer string."""
        context_chunks = await self._get_full_pipeline_response(user_query)
        return await self.generate_answer(user_query, context_chunks)

    async def answer_stream(self, user_query: str):
        """Runs the full pipeline and yields the answer as a stream of text."""
        context_chunks = await self._get_full_pipeline_response(user_query)
        async for token in self.generate_answer_stream(user_query, context_chunks):
            yield token
