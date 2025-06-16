import json
import os
import time
import re

pptx_path = "data/processed/pptx_chunks.json"
video_path = "data/processed/video_chunks.json"
clueso_path = "data/processed/clueso_grouped_blocks.json"
out_path = "data/processed/all_chunks_normalized.json"

STEP_PATTERN = re.compile(r"^step\s*1\b", re.IGNORECASE)

def normalize_pptx(slides):
    norm = []
    for slide in slides:
        text = "\n".join(block["content"] for block in slide.get("blocks", []) if block["type"] == "text")
        slide_number = slide.get("slide_number", None)
        norm.append({
            "chunk_id": slide["chunk_id"] if "chunk_id" in slide else slide_number,
            "source": "2023 Reporting Portal Trunk- Manager Guide",
            "source_detail": slide['source_detail'],
            "text": text,
            "blocks": slide.get("blocks", []),
            "metadata": {
                "slide_number": slide_number,
                "pptx_file": os.path.basename(pptx_path)
            }
        })
    return norm

def normalize_video(chunks):
    norm = []
    for i, chunk in enumerate(chunks):
        start = chunk.get("start")
        end = chunk.get("end")
        start_str = chunk.get("metadata", {}).get("start_str") or chunk.get("start_str")
        end_str = chunk.get("metadata", {}).get("end_str") or chunk.get("end_str")

        def fmt(secs):
            return time.strftime("%M:%S", time.gmtime(secs)) if secs is not None else "?"

        if not start_str and start is not None:
            start_str = fmt(start)
        if not end_str and end is not None:
            end_str = fmt(end)

        if start_str and end_str:
            source_detail = f"timestamp {start_str}–{end_str}"
        else:
            source_detail = f"token_chunk_{i+1}"

        norm.append({
            "chunk_id": i + 1,
            "source": "Org Vitality Reporting Portal Guide Video",
            "source_detail": source_detail,
            "text": chunk["text"] if "text" in chunk else chunk,
            "blocks": [],
            "metadata": {
                "orig_index": i,
                "video_file": os.path.basename(video_path),
                "start": start,
                "end": end,
                "start_str": start_str,
                "end_str": end_str
            }
        })
    return norm

def normalize_clueso(raw_data):
    # Flatten any accidental nesting
    flat_blocks = [block for group in raw_data for block in group]

    grouped = []
    current_group = []

    for block in flat_blocks:
        if block["type"] == "text" and STEP_PATTERN.match(block.get("text", "").strip()):
            if current_group:
                grouped.append(current_group)
                current_group = []
        current_group.append(block)

    if current_group:
        grouped.append(current_group)

    # Normalize each grouped block
    norm = []
    for i, group in enumerate(grouped):
        text = "\n".join(
            b.get("text", b.get("content", "")) for b in group if b["type"] == "text"
        )
        norm.append({
            "chunk_id": i + 1,
            "source": "Org Vitality Reporting Portal Guide",
            "source_detail": f"step_group_{i+1}",
            "text": text,
            "blocks": group,
            "metadata": {
                "num_blocks": len(group)
            }
        })

    return norm

# -----------------------
# Execute full normalization
# -----------------------

all_norm = []

if os.path.exists(pptx_path):
    with open(pptx_path) as f:
        all_norm += normalize_pptx(json.load(f))

if os.path.exists(video_path):
    with open(video_path) as f:
        all_norm += normalize_video(json.load(f))

if os.path.exists(clueso_path):
    with open(clueso_path) as f:
        all_norm += normalize_clueso(json.load(f))

# Reassign chunk IDs globally
for i, c in enumerate(all_norm):
    c["chunk_id"] = i + 1

with open(out_path, "w") as f:
    json.dump(all_norm, f, indent=2)

print(f"✅ Done! Merged and normalized {len(all_norm)} chunks to {out_path}")
