import whisper
import tiktoken
import json
import time

VIDEO_PATH = "data/raw/video.mp4"
OUTPUT_TXT = "data/processed/video_transcript.txt"
OUTPUT_JSON = "data/processed/video_chunks.json"
CHUNK_DURATION = 30  # seconds
MODEL = "gpt-4o"

def format_timestamp(seconds):
    return time.strftime('%M:%S', time.gmtime(seconds))

def chunk_segments_by_time(segments, chunk_duration=CHUNK_DURATION):
    chunks = []
    buffer = []
    start_time = None
    for segment in segments:
        if not buffer:
            start_time = segment["start"]
        buffer.append(segment)
        end_time = segment["end"]

        if end_time - start_time >= chunk_duration:
            chunk_text = " ".join(seg["text"] for seg in buffer)
            chunks.append({
                "text": chunk_text,
                "start": start_time,
                "end": end_time,
                "segments": buffer.copy()
            })
            buffer = []
            start_time = None

    # Last chunk
    if buffer:
        end_time = buffer[-1]["end"]
        chunk_text = " ".join(seg["text"] for seg in buffer)
        chunks.append({
            "text": chunk_text,
            "start": start_time,
            "end": end_time,
            "segments": buffer.copy()
        })

    return chunks

# Load the Whisper model
model = whisper.load_model("small")

# Transcribe
result = model.transcribe(VIDEO_PATH, verbose=True)

# Save raw transcript
with open(OUTPUT_TXT, "w") as f:
    f.write(result["text"])

# Chunk the transcript by ~30 second segments
chunks = chunk_segments_by_time(result["segments"])

# Save as JSON
chunk_objs = []
for i, chunk in enumerate(chunks):
    start_str = format_timestamp(chunk["start"])
    end_str = format_timestamp(chunk["end"])
    chunk_objs.append({
        "chunk_id": i + 1,
        "source": "video",
        "source_detail": f"timestamp {start_str}–{end_str}",
        "text": chunk["text"],
        "blocks": [],
        "metadata": {
            "start": chunk["start"],
            "end": chunk["end"],
            "start_str": start_str,
            "end_str": end_str,
            "video_file": VIDEO_PATH
        }
    })

with open(OUTPUT_JSON, "w") as f:
    json.dump(chunk_objs, f, indent=2)

print(f"Transcription complete! Transcript saved to {OUTPUT_TXT}")
print(f"Chunked output saved to {OUTPUT_JSON} ({len(chunk_objs)} chunks)")
