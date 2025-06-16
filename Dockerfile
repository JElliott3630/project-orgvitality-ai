# Step 1: Use the official Python slim image
FROM python:3.11-slim

# Step 2: Set up the working directory in the container
WORKDIR /app

# Step 3: Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the requirements file and install Python packages
# This is done separately to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 5: Install Playwright browsers if needed for other scripts
RUN playwright install --with-deps

# Step 6: Copy the entire application source code into the container
# This creates the /app/src directory structure
COPY src/ ./src

# Step 7: Expose the port the app runs on
EXPOSE 8000

# Step 8: Define the command to run the application
# This now correctly points to the app object within the src package
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
