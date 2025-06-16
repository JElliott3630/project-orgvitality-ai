import openai
import json
import yaml
import logging
from sentence_transformers import CrossEncoder
from config import OPENAI_API_KEY, PROMPT_PATH

# Setup logging (do this only once at app entry if you want to customize further)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

def load_prompts(path=PROMPT_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class RagPipeline:
    def __init__(self, vector_store, prompts=None, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2", rebuild_vectorstore=False):
        self.vector_store = vector_store

        # Only rebuild if requested or if vector store is empty
        if rebuild_vectorstore or not self.vector_store.is_built():
            logging.info("Building or rebuilding the vector store...")
            self.vector_store.build_from_json()
        else:
            logging.info("Vector store already built, skipping rebuild.")

        self.prompts = prompts or load_prompts()
        openai.api_key = OPENAI_API_KEY
        self.cross_encoder = CrossEncoder(reranker_model)

    def expand_query(self, user_query):
        sys_prompt = self.prompts["query_expansion"]["system_prompt"]
        user_prompt = self.prompts["query_expansion"]["user_prompt_template"].format(query_text=user_query)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        expanded = response.choices[0].message.content.strip()
        try:
            subqueries = json.loads(expanded)
        except Exception as e:
            logging.warning("Query expansion failed, using original query. Error: %s", str(e))
            subqueries = [user_query]
        return subqueries

    def retrieve(self, queries, k=8):
        if isinstance(queries, str):
            queries = [queries]
        retrieved_chunks = []
        for q in queries:
            results = self.vector_store.query(q, n_results=k)
            docs = results["documents"][0] if "documents" in results else results["documents"]
            metadatas = results.get("metadatas", [{}]*len(docs))[0] if "metadatas" in results else [{}]*len(docs)
            for doc, meta in zip(docs, metadatas):
                retrieved_chunks.append({"text": doc, "metadata": meta, "query": q})
        return retrieved_chunks

    def rerank(self, user_query, retrieved_chunks, top_n=6):
        pairs = [(user_query, chunk["text"]) for chunk in retrieved_chunks]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        scored_chunks = list(zip(retrieved_chunks, scores))
        reranked = [chunk for chunk, score in sorted(scored_chunks, key=lambda x: x[1], reverse=True)[:top_n]]
        return reranked

    def generate_answer(self, user_query, reranked_chunks, stream=False):
        sys_prompt = self.prompts["answer_generation"]["system_prompt"]

        def format_chunk_for_context(chunk):
            label_parts = []
            if "source" in chunk['metadata']:
                label_parts.append(f"Source: {chunk['metadata']['source']}")
            if "source_detail" in chunk['metadata'] and chunk['metadata']["source_detail"]:
                label_parts.append(str(chunk['metadata']["source_detail"]))
            label = ", ".join(label_parts)
            return f"[{label}]\\n{chunk['text']}"

        context_block = "\\n\\n".join([format_chunk_for_context(chunk) for chunk in reranked_chunks])
        user_prompt = self.prompts["answer_generation"]["user_prompt_template"].format(
            query_text=user_query,
            context=context_block
        )
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=stream
        )
        
        if stream:
            return response
        else:
            answer = response.choices[0].message.content.strip()
            return answer

    def answer(self, user_query):
        subqueries = self.expand_query(user_query)
        retrieved = self.retrieve(subqueries)
        reranked = self.rerank(user_query, retrieved)
        answer = self.generate_answer(user_query, reranked, stream=False)
        return answer

    def answer_stream(self, user_query):
        """Yields the LLM's response as a stream of chunks."""
        subqueries = self.expand_query(user_query)
        retrieved = self.retrieve(subqueries)
        reranked = self.rerank(user_query, retrieved)
        
        stream = self.generate_answer(user_query, reranked, stream=True)
        
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content