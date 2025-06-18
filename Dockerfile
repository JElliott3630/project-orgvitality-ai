# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install only essential system dependencies
# Removed: tesseract-ocr, tesseract-ocr-eng, libgl1-mesa-glx, libgomp1, build-essential, curl, gnupg
# These were primarily for Playwright, Tesseract, or general build tools not always needed at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Add any *truly essential* system packages here if your remaining Python libs require them.
    # For a typical FastAPI app with pure Python dependencies, often very few are needed beyond base 'slim' image.
    # Example for some database drivers: libpq-dev for psycopg2-binary (PostgreSQL)
    # Example for image processing if you still use Pillow: libjpeg-dev zlib1g-dev
    # For now, we'll keep it minimal.
    && rm -rf /var/lib/apt/lists/*

# Copy the updated requirements file into the container at /app
# This assumes you have already removed 'playwright', 'openai-whisper', 'pytesseract', 'python-pptx', 'lxml'
COPY requirements.txt .

# Install any needed packages specified in the now smaller requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Ensure the data directory exists and has appropriate permissions (for processed JSON, as requested)
# This step is crucial if your app reads from /app/data during runtime.
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Expose the port the app runs on
EXPOSE 8000

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]