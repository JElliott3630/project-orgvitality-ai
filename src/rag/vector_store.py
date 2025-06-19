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
        """
        Initializes the Pinecone client and connects to or creates a user-specific index.

        Args:
            user_id (str): A unique identifier for the user, used to create a dedicated Pinecone index.
        """
        self.user_id = user_id
        # Index names must be lowercase and cannot contain underscores for Pinecone
        # self.index_name = f"index-{self.user_id}".lower().replace("_", "-")
        self.index_name = "index-orgvitality-default"
        self._pc_client = Pinecone(api_key=config.PINECONE_API_KEY)
        self._initialize_index()

    def _initialize_index(self):
        """
        Checks if the Pinecone index for the current user exists. If not, it creates it.
        Otherwise, it connects to the existing index.
        """
        if self.index_name not in self._pc_client.list_indexes().names():
            print(f"Creating Pinecone index: {self.index_name}")
            self._pc_client.create_index(
                name=self.index_name,
                dimension=config.EMBEDDING_DIMENSION, # Use dimension from config
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=config.PINECONE_REGION # Use region from config
                )
            )
            print(f"Pinecone index '{self.index_name}' created.")
        else:
            print(f"Connecting to existing Pinecone index: {self.index_name}")

        self.index = self._pc_client.Index(self.index_name)
        # It's good practice to print index stats for confirmation
        print("Pinecone index stats:", self.index.describe_index_stats())

    def upsert_vectors(self, vectors: list[dict]):
        """
        Upserts (inserts or updates) a list of vectors into the Pinecone index.

        Args:
            vectors (list[dict]): A list of dictionaries, where each dict represents a vector
                                  with 'id', 'values', and 'metadata'.
        """
        if not vectors:
            print("No vectors provided for upsert. Skipping operation.")
            return
        

        print(f"Upserting {len(vectors)} vectors to Pinecone index '{self.index_name}'...")
        # Pinecone's upsert can take vectors in batches; consider batching for very large lists
        self.index.upsert(vectors=vectors)
        print("✅ Upsert complete.")

    def query_vectors(self, query_embedding: list[float], top_k: int, metadata_filter: dict = None) -> list[dict]:
        """
        Queries the Pinecone index with a given embedding and retrieves the top_k most similar vectors.

        Args:
            query_embedding (list[float]): The embedding of the query.
            top_k (int): The number of top relevant vectors to retrieve.
            metadata_filter (dict, optional): A dictionary for metadata filtering. Defaults to None.

        Returns:
            list[dict]: A list of dictionaries, each representing a retrieved chunk
                        with its content and metadata.
        """
        if not query_embedding:
            print("Query embedding is empty. Cannot perform query.")
            return []

        print(f"Querying Pinecone index '{self.index_name}' with top_k={top_k}...")
        query_response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True, # Crucial to retrieve the stored text and other metadata
            filter=metadata_filter # Apply metadata filters if provided
        )

        retrieved_chunks = []
        for match in query_response.matches:
            # Ensure 'text' and 'source' are present in metadata for robust retrieval
            page_content = match.metadata.get("text", "")
            source = match.metadata.get("source", "N/A")
            page = match.metadata.get("page", "N/A") # Assuming page number is also stored

            retrieved_chunks.append({
                "page_content": page_content,
                "metadata": {
                    "source": source,
                    "page": page
                }
            })
        print(f"Retrieved {len(retrieved_chunks)} chunks from Pinecone.")
        return retrieved_chunks