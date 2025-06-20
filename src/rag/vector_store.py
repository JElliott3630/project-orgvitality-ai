import os
from pinecone import Pinecone, ServerlessSpec
from src import config
import logging


class PineconeVectorStore:
    """
    Manages Pinecone index operations including initialization, upserting, and querying.
    This class centralizes all direct interactions with the Pinecone vector database.
    """

    def __init__(self, user_id: str):
        """Connect to (or create) the single‑tenant OrgVitality index."""
        self.user_id = user_id
        self.index_name = "index-orgvitality-default"
        self._pc_client = Pinecone(api_key=config.PINECONE_API_KEY)
        self._initialize_index()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initialize_index(self):
        if self.index_name not in self._pc_client.list_indexes().names():
            print(f"Creating Pinecone index: {self.index_name}")
            self._pc_client.create_index(
                name=self.index_name,
                dimension=config.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=config.PINECONE_REGION),
            )
            print(f"Pinecone index '{self.index_name}' created.")
        else:
            print(f"Connecting to existing Pinecone index: {self.index_name}")

        self.index = self._pc_client.Index(self.index_name)
        print("Pinecone index stats:", self.index.describe_index_stats())

    # ---- new: safe batched upsert -----------------------------------

    @staticmethod
    def _chunk(vectors: list[dict], size: int):
        for i in range(0, len(vectors), size):
            yield vectors[i : i + size]

    def _upsert_batched(self, vectors: list[dict], batch: int = 80):
        """Internal helper that upserts in <=2 MB chunks (≈80 vectors)."""
        for slice_ in self._chunk(vectors, batch):
            self.index.upsert(vectors=slice_)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_vectors(self, vectors: list[dict]):
        """Upsert vectors, automatically batching if payload is large."""
        if not vectors:
            print("No vectors provided for upsert. Skipping operation.")
            return

        print(f"Upserting {len(vectors)} vectors to Pinecone index '{self.index_name}'…")
        if len(vectors) >= 80:
            self._upsert_batched(vectors)
        else:
            self.index.upsert(vectors=vectors)
        print("✅ Upsert complete.")

    # ------------------------------------------------------------------

    def query_vectors(
        self,
        query_embedding: list[float],
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Semantic search helper (unchanged)."""
        if not query_embedding:
            print("Query embedding is empty. Cannot perform query.")
            return []

        print(f"Querying Pinecone index '{self.index_name}' with top_k={top_k}…")
        query_response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter,
        )

        chunks = []
        for match in query_response.matches:
            chunks.append(
                {
                    "page_content": match.metadata.get("text", ""),
                    "metadata": {
                        "source": match.metadata.get("source", "N/A"),
                        "page": match.metadata.get("page", "N/A"),
                    },
                }
            )
        print(f"Retrieved {len(chunks)} chunks from Pinecone.")
        return chunks
