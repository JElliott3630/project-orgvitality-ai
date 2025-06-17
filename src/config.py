import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv(override=True)

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Supabase
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Secure backend auth

# --- Base Paths ---
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
DATA_DIR = os.path.join(PROJ_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CHROMA_DIR = os.path.join(DATA_DIR, "chromadb")
EVAL_DATA_DIR = os.path.join(DATA_DIR, "chatbot_eval_questions")
COLLECTION_NAME = "orgvitality_chunks"
EMBED_MODEL = "text-embedding-3-small"

# --- Source File Paths ---
PPTX_INPUT_PATH = os.path.join(RAW_DATA_DIR, "2023 Reporting Portal Trunk- Manager Guide  -  Read-Only .pptx")
VIDEO_INPUT_PATH = os.path.join(RAW_DATA_DIR, "OrgVitality Reporting Portal Demo.mp4")
CLUESO_INPUT_PATH = os.path.join(RAW_DATA_DIR, "clueso_raw.csv")

# --- Processed File Paths ---
PPTX_OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "pptx_chunks.json")
VIDEO_TRANSCRIPT_PATH = os.path.join(PROCESSED_DATA_DIR, "video_transcript.txt")
VIDEO_CHUNKS_PATH = os.path.join(PROCESSED_DATA_DIR, "video_chunks.json")
CLUESO_OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "clueso_step_chunks.json")
NORMALIZED_CHUNKS_PATH = os.path.join(PROCESSED_DATA_DIR, "all_chunks_normalized.json")
PROCESSED_JSON = NORMALIZED_CHUNKS_PATH

# --- Source Names ---
SOURCE_NAMES = {
    "pptx": "2023 Reporting Portal Trunk- Manager Guide",
    "video": "OrgVitality Reporting Portal Demo",
    "clueso": "Clueso Steps"
}

# --- Prompt Configuration ---
PROMPT_PATH = os.path.join(PROJ_ROOT, "src", "prompts.yml")

# --- Evaluation Data ---
EVAL_QUESTIONS_PATH = os.path.join(EVAL_DATA_DIR, "OV Provided Questions 601578c63b2647eb93941d02c0f67a58.csv")
