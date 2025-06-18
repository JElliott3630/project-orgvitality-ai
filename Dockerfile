# Use an official Python runtime as a parent image
# Using -slim-buster ensures a stable Debian base for apt-get
FROM python:3.11-slim-buster

# Set environment variables for non-interactive apt-get and Python
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1    

# Set the working directory in the container
WORKDIR /app

# Install only essential system dependencies and clean up apt cache immediately.
# Combining RUN commands reduces the number of layers.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # Add any *truly essential* system packages here if your remaining Python libs require them.
    # For common ML/data science libraries, you might need:
    # libgl1-mesa-glx (for some image/graphing libraries, often pulls from NVIDIA),
    # libgomp1 (OpenMP, needed by numpy/scipy for faster computation)
    # If you encounter runtime errors related to missing shared libraries, add them here.
    # For now, keeping it minimal as before.
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy the updated requirements file (now without 'torch' and 'typing_extensions') into the container at /app
COPY requirements.txt .

# Explicitly install typing-extensions before torch to resolve potential naming conflicts
# and ensure a compatible version is available from the standard PyPI index.
# torch requires >=4.10.0, so 4.14.0 (or newer) should be fine.
RUN pip install --no-cache-dir typing-extensions==4.14.0

# Install PyTorch CPU-only. This is CRITICAL for smaller image size on Cloud Run.
# Ensure this URL is correct for your desired torch version and CPU.
# Now, torch should find typing-extensions already installed.
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies from requirements.txt.
# --no-deps prevents pip from trying to reinstall torch (or its GPU deps) as a transitive dependency.
# It also ensures that typing_extensions is not processed again by pip from requirements.txt,
# as it should have been handled by the explicit install and torch's dependency resolution.
RUN pip install --no-cache-dir -r requirements.txt --no-deps

# Copy the rest of your application code
# It's important to put this after pip install to leverage Docker caching.
# If only your code changes, this layer and subsequent layers will be rebuilt,
# but the slower pip install layer can be reused.
COPY . .

# Ensure the data directory exists and has appropriate permissions
# This is for /app/data, which we've excluded from .dockerignore for copying into the image,
# but if your app expects it to exist for temporary files or output, it's fine.
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Expose the port the app runs on (primarily for documentation)
EXPOSE 8080 

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
