import json
import asyncio
import logging
from langfuse.openai import openai
from src.config import (
    PINECONE_API_KEY, PINECONE_REGION, EMBEDDING_DIMENSION,
    OPENAI_API_KEY, DEFAULT_PINECONE_USER_ID, EMBEDDING_MODEL # Added EMBEDDING_MODEL to import
)
from src.rag.vector_store import PineconeVectorStore # Your PineconeVectorStore class

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s"
)

async def generate_embeddings(texts: list[str], client: openai.AsyncClient, model: str) -> list[list[float]]:
    """Generates embeddings for a list of texts using OpenAI's API."""
    try:
        response = await client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    except openai.APIError as e:
        logging.error(f"Error generating embeddings: {e}")
        raise

async def populate_pinecone_db(json_file_path: str, user_id: str, embedding_model: str):
    """
    Reads chunks from a JSON file, generates embeddings, and upserts them into Pinecone.
    """
    if not PINECONE_API_KEY or not PINECONE_REGION:
        logging.error("Pinecone API key or region is not set in environment variables.")
        return

    # Initialize OpenAI AsyncClient
    openai_client = openai.AsyncClient(api_key=OPENAI_API_KEY)

    # Initialize Pinecone Vector Store for the given user_id
    pinecone_store = PineconeVectorStore(user_id=user_id)

    logging.info(f"Loading data from {json_file_path}...")
    try:
        with open(json_file_path, 'r') as f:
            chunks = json.load(f) #
    except FileNotFoundError:
        logging.error(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error: Invalid JSON format in {json_file_path}")
        return

    logging.info(f"Loaded {len(chunks)} chunks. Generating embeddings and upserting to Pinecone...")

    # Batching for embedding generation and upserting is crucial for performance
    batch_size = 100 # Adjust based on your API rate limits and memory
    for i in range(0, len(chunks), batch_size):
        raw_batch_chunks = chunks[i:i + batch_size]

        valid_chunks_in_batch = []
        texts_to_embed = []
        # --- CRITICAL FILTERING LOGIC ---
        for chunk in raw_batch_chunks:
            chunk_text = chunk.get("text")
            # Ensure it's a string and not just whitespace
            if isinstance(chunk_text, str) and chunk_text.strip():
                valid_chunks_in_batch.append(chunk)
                texts_to_embed.append(chunk_text)
            else:
                logging.warning(f"Skipping invalid or empty text for chunk ID: {chunk.get('chunk_id', 'N/A')} in batch {i}-{i+batch_size}.")
        # --- END CRITICAL FILTERING LOGIC ---

        if not texts_to_embed:
            logging.info(f"No valid texts to embed in batch {i}-{i+batch_size}. Skipping.")
            continue # Skip to the next batch if no valid texts

        # Generate embeddings for only the valid texts
        try:
            embeddings_batch = await generate_embeddings(texts_to_embed, openai_client, embedding_model)
        except Exception as e:
            logging.error(f"Skipping batch {i}-{i+len(valid_chunks_in_batch)} due to embedding error: {e}")
            continue

        vectors_to_upsert = []
        # Iterate over valid_chunks_in_batch which corresponds to embeddings_batch
        for j, chunk in enumerate(valid_chunks_in_batch):
            chunk_text = chunk.get("text", "") # Get the text again for preparing metadata

            # Ensure 'id' is a string and unique. Using a combination of user_id, chunk_id, batch_index, and item_index
            vector_id = f"{user_id}-chunk-{chunk.get('chunk_id', '')}-{i}-{j}"
            # Fallback if chunk_id is entirely missing or not suitable for ID
            if not chunk.get('chunk_id'):
                vector_id = f"{user_id}-batch-{i}-idx-{j}" # More unique fallback ID

            # Prepare metadata for Pinecone. Ensure it's flat and JSON-serializable.
            metadata = {
                "source": chunk.get("source", "N/A"),
                "source_detail": chunk.get("source_detail", "N/A"),
                "text": chunk_text # Store the original text to retrieve it later
            }
            # Add other relevant metadata fields from your JSON if needed
            if 'metadata' in chunk: # If there's nested metadata in your original JSON
                for k, v in chunk['metadata'].items():
                    # Pinecone accepts basic types, lists, and dicts in metadata directly
                    if isinstance(v, (str, int, float, bool, list, dict)):
                        metadata[k] = v
                    else:
                        # Convert complex types to string to ensure compatibility
                        metadata[k] = str(v)

            vectors_to_upsert.append({
                "id": vector_id,
                "values": embeddings_batch[j],
                "metadata": metadata
            })

        if vectors_to_upsert:
            await asyncio.to_thread(pinecone_store.upsert_vectors, vectors_to_upsert)
            logging.info(f"Upserted {len(vectors_to_upsert)} vectors for batch {i}-{i+len(vectors_to_upsert)}.")
        else:
            logging.warning(f"No valid vectors to upsert for batch {i}-{i+len(valid_chunks_in_batch)}.")


    logging.info(f"Pinecone population complete for user '{user_id}'.")

if __name__ == "__main__":
    # Ensure your environment variables are set before running this script:
    # export OPENAI_API_KEY='your_openai_api_key'
    # export PINECONE_API_KEY='your_pinecone_api_key'
    # export PINECONE_REGION='your_pinecone_region' (e.g., 'us-east-1')

    # Path to your all_chunks_normalized.json file
    data_file = "data/processed/all_chunks_normalized.json" # Adjust path if needed
    # Make sure this matches the EMBEDDING_MODEL in src/config.py
    embedding_model_name = EMBEDDING_MODEL # Use EMBEDDING_MODEL from config

    asyncio.run(populate_pinecone_db(
        json_file_path=data_file,
        user_id=DEFAULT_PINECONE_USER_ID,
        embedding_model=embedding_model_name
    ))