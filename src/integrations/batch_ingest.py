"""Batch Dropbox → Pinecone ingestion (single-tenant).

Traverses every .pdf/.docx/.xlsx/.pptx in the shared Dropbox folder, parses &
embeds all chunks, then upserts the resulting vectors into a single
`index-orgvitality-default` Pinecone index.

Environment variables
---------------------
DROPBOX_TOKEN     – Dropbox API token
OPENAI_API_KEY    – OpenAI key for embeddings
PINECONE_API_KEY  – Pinecone key (loaded in config.py)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, List, Dict

import dropbox

# --- project imports --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # project root
SRC_DIR      = PROJECT_ROOT / "src"                 # src package directory
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.ingestion import DocumentIngestor
from core.embeddings import OpenAIEmbedding
from core.schema import DocumentBatch
from rag.vector_store import PineconeVectorStore
from config import DROPBOX_TOKEN  # <-- token now comes from central config

# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("batch_ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- hard‑coded settings -------------------------------------------
ROOT_FOLDER = "/Knowledge Base to give SFAI"   # Dropbox root folder
ALLOWED_EXT = {".pdf", ".docx", ".xlsx", ".pptx"}
USER_ID     = "orgvitality"                    # single‑tenant store
# ---------------------------------------------------------------------------


def list_files(dbx: dropbox.Dropbox, root_folder: str) -> Iterable[dropbox.files.FileMetadata]:
    """Yield every FileMetadata under *root_folder* (recursive)."""
    queue: List[str] = [root_folder]
    while queue:
        path = queue.pop()
        result = dbx.files_list_folder(path, recursive=False)
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FolderMetadata):
                queue.append(entry.path_lower)
            elif isinstance(entry, dropbox.files.FileMetadata):
                yield entry
        while result.has_more:
            result = dbx.files_list_folder_continue(result.cursor)
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FileMetadata):
                    yield entry


# ---- embeddings → vector objects ----------------------------------------

EMB_BATCH = 80   # 80 × 500‑token chunks keeps payload <512 kB
UP_BATCH  = 80   # 80 vectors ≈ 1.2 MB < Pinecone 2 MB limit


def _batch_to_vectors(batch: DocumentBatch, embedder: OpenAIEmbedding) -> List[Dict]:
    texts, vectors = [c.text for c in batch.chunks], []
    for i in range(0, len(texts), EMB_BATCH):
        vectors.extend(embedder.embed_texts(texts[i : i + EMB_BATCH]))

    return [
        {
            "id": c.id,
            "values": v,
            "metadata": {
                "text": c.text,
                "source": c.source,
                "location": c.location,
                "doc_id": batch.doc_id,
            },
        }
        for c, v in zip(batch.chunks, vectors)
    ]


# ---- ingestion & upsert --------------------------------------------------

def ingest_file(
    dbx: dropbox.Dropbox,
    entry: dropbox.files.FileMetadata,
    ingestor: DocumentIngestor,
    embedder: OpenAIEmbedding,
    store: PineconeVectorStore,
):
    LOGGER.info("Processing %s", entry.path_display)
    try:
        _, resp = dbx.files_download(entry.path_lower)
        batch: DocumentBatch = ingestor.ingest(resp.content, entry.name)
        vectors = _batch_to_vectors(batch, embedder)
        for i in range(0, len(vectors), UP_BATCH):
            store.upsert_vectors(vectors[i : i + UP_BATCH])
        LOGGER.info("  ↳ upserted %d chunks", len(batch.chunks))
    except Exception as err:
        LOGGER.exception("Failed to process %s: %s", entry.path_display, err)


# ---- main ----------------------------------------------------------------

def main() -> None:
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)

    embedder = OpenAIEmbedding()
    ingestor = DocumentIngestor()
    store    = PineconeVectorStore(user_id=USER_ID)

    TARGETS = {
        "Creating the Vital Organization.pdf",
        "How Performance Management Impacts Engagement.pdf",
    }

    for entry in list_files(dbx, ROOT_FOLDER):
        if entry.name in TARGETS and Path(entry.name).suffix.lower() in ALLOWED_EXT:
            ingest_file(dbx, entry, ingestor, embedder, store)


if __name__ == "__main__":
    main()
