# 1. Use an official NVIDIA CUDA image
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

# Set the working directory
WORKDIR /app

# Prevent prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 2. Install Python, pip, Git, and essential build tools
#    - python3.11-dev is crucial for C header files
#    - build-essential provides C/C++ compilers
#    - git is needed for some pip installations
RUN apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git && \
    rm -rf /var/lib/apt/lists/*

# 3. Upgrade pip and copy requirements file
COPY src/requirements.txt src/requirements.txt

# 4. Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install -r src/requirements.txt

# 5. Copy the rest of your application code
COPY src/ src/
ENV PYTHONPATH /app

# Set environment variables for Streamlit and Python
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Expose the port that Streamlit runs on
EXPOSE 8501

# 6. The command to run when the container starts
CMD ["streamlit", "run", "src/inventory_dashboard_streamlit.py"]
