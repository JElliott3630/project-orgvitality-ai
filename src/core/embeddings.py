import os
from langfuse.openai import openai
import backoff   # make sure already installed

class OpenAIEmbedding:
    """
    Thin wrapper around the OpenAI v1 client (`pip install openai>=1.0`).

    Usage:
        embedder = OpenAIEmbedding(model="text-embedding-3-small")
        vectors  = embedder.embed_texts(["hello", "world"])
    """
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model  = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # automatic exponential back-off on rate-limit / transient errors
    @backoff.on_exception(backoff.expo,
                          (openai.RateLimitError, openai.APIError),
                          max_tries=5)
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Returns one embedding vector per text, in the same order.
        """
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        # v1 returns resp.data[i].embedding
        return [d.embedding for d in resp.data]
