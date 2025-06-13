import chromadb
from chromadb.utils import embedding_functions
import logging

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    OPENAI_API_KEY,
    PROCESSED_JSON
)

# Set up logging once (at INFO level, minimal)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

class VectorStore:
    def __init__(
        self,
        persist_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBED_MODEL,
        openai_api_key=OPENAI_API_KEY
    ):
        logging.info("Initializing VectorStore (dir: %s, collection: %s, embed_model: %s)", persist_dir, collection_name, embedding_model)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.openai_api_key = openai_api_key

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedder = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name=self.embedding_model
        )
        if self.collection_name in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(self.collection_name, embedding_function=self.embedder)
        else:
            self.collection = self.client.create_collection(self.collection_name, embedding_function=self.embedder)

    def is_built(self):
        """Returns True if the collection exists and has any documents."""
        return self.collection.count() > 0
    
    def clean_metadata(self, meta):
        return {k: v if v is not None else "" for k, v in meta.items() if v is not None}

    def build_from_chunks(self, chunks, rebuild=False):
        if rebuild and self.collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(self.collection_name, embedding_function=self.embedder)
        logging.info("Adding %d chunks to collection...", len(chunks))

        docs = []
        ids = []
        metadatas = []
        for chunk in chunks:
            meta = {
                "source": chunk["source"],
                "source_detail": chunk["source_detail"],
                "chunk_id": chunk["chunk_id"],
            }
            # Flatten metadata if exists and is a dict
            if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                meta.update(chunk["metadata"])
            meta = self.clean_metadata(meta)
            docs.append(str(chunk["text"]))    # Make sure it's a string
            ids.append(f"{chunk['source']}_{chunk['chunk_id']}")
            metadatas.append(meta)

        # Filter out any empty docs (to avoid OpenAI error)
        filtered = [
            (d, i, m) for d, i, m in zip(docs, ids, metadatas)
            if d.strip()
        ]
        if not filtered:
            raise ValueError("No valid (non-empty) docs to index.")

        docs, ids, metadatas = zip(*filtered)

        self.collection.add(
            documents=list(docs),
            metadatas=list(metadatas),
            ids=list(ids)
        )


    def build_from_json(self, json_path=PROCESSED_JSON, rebuild=True):
        import json
        with open(json_path) as f:
            chunks = json.load(f)
        self.build_from_chunks(chunks, rebuild=rebuild)

    def query(self, text, n_results=5):
        return self.collection.query(
            query_texts=[text],
            n_results=n_results
        )

    def count(self):
        cnt = self.collection.count()
        return cnt

    def reset(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(self.collection_name, embedding_function=self.embedder)
