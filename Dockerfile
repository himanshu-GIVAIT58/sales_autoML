# 1. Use an official NVIDIA CUDA image that matches your driver's CUDA version (12.2)
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

# Set the working directory for all subsequent commands
WORKDIR /app

# Prevent prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 2. Install Python 3.11 and pip
RUN apt-get update && \
    apt-get install -y python3.11 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 3. Copy only the requirements file first to leverage Docker layer caching.
COPY src/requirements.txt src/requirements.txt

# 4. Install Python dependencies.
#    This will now install GPU-compatible versions of libraries if specified correctly.
RUN pip3 install --no-cache-dir -r src/requirements.txt

# 5. Now, copy the rest of your application code.
COPY src/ src/
ENV PYTHONPATH /app

# Set environment variables for Streamlit and Python
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Expose the port that Streamlit runs on
EXPOSE 8501

# 6. The command to run when the container starts.
CMD ["streamlit", "run", "src/inventory_dashboard_streamlit.py"]
