import os
import logging
os.environ["TQDM_DISABLE"] = "1"
os.environ["OPENAI_LOG"] = "error"

import csv
from rag.vector_store import VectorStore
from rag.rag_pipeline import RagPipeline



for name in ["httpx", "openai", "urllib3"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

# Adjust these if your config is elsewhere
VECTOR_STORE = VectorStore()
rag = RagPipeline(vector_store=VECTOR_STORE)

csv_path = "data/chatbot_eval_questions/OV Provided Questions 601578c63b2647eb93941d02c0f67a58.csv"



# Read first 10 'A' class questions
questions = []
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row.get("Class", "").strip().upper() == "A" and row.get("Question", "").strip():
            questions.append(row["Question"])
        if len(questions) == 3:
            break

print(f"Loaded {len(questions)} 'A' class questions.")



for i, q in enumerate(questions, 1):
    # retrieved = VECTOR_STORE.query(q)
    # documents = retrieved['documents'][0]
    # metadatas = retrieved['metadatas'][0]

    # print(f"Retrieved {len(documents)} chunks for Q{i+1}:")
    # for doc, meta in zip(documents, metadatas):
    #     print(f"- {doc[:80]}..., {meta.get('source_detail','')}]")

    print(f"\n=== Q{i}: {q} ===")
    answer = rag.answer(q)
    print(f"\nQ{i}: {q}")
    print(f"A{i}: {answer}\n")
