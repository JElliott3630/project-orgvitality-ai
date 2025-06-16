# Step 1: Use the official Python slim image for a smaller footprint
FROM python:3.11-slim

# Step 2: Set up the working directory in the container
WORKDIR /app

# Step 3: Install system dependencies
# tesseract-ocr is required for the pytesseract library
# libgl1 is a common dependency for headless browsers (used by Playwright)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the requirements file and install Python packages
# This step is done separately to leverage Docker's layer caching
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 5: Install Playwright browsers and their dependencies
# This is needed for the data scraping script
RUN playwright install --with-deps

# Step 6: Copy the application source code into the container
COPY src/ ./src

# Step 7: Expose the port the app runs on
EXPOSE 8000

# Step 8: Define the command to run the application
# We run the uvicorn server, pointing to the FastAPI app instance in rag_api.py
# The host is set to 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "src.rag_api:app", "--host", "0.0.0.0", "--port", "8000"]