# Use an official Python runtime as a parent image
FROM python:3.11-slim 

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by playwright and tesseract
# Install other necessary packages like build-base, tesseract-ocr
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libgomp1 \
    build-essential \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# This step will install the 'playwright' Python package, making the 'playwright' command available.
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (chromium) and their dependencies if you use playwright for web scraping
# This command must run AFTER playwright Python package is installed via pip
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN playwright install --with-deps chromium

# Copy the rest of your application code
COPY . .

# Ensure the data directory exists and has appropriate permissions (for processed JSON if needed)
# ChromaDB will store its data on the remote GCE server, but your app might need this for input JSON.
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Expose the port the app runs on
EXPOSE 8000

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]