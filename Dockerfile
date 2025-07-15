# Use an official Python image as the base image
FROM python:3.11-slim

# Set the working directory for all subsequent commands
WORKDIR /app

# 1. Copy only the requirements file first to leverage Docker layer caching.
COPY src/requirements.txt src/requirements.txt

# 2. Install Python dependencies.
RUN pip install --default-timeout=100 --no-cache-dir -r src/requirements.txt

# 3. Now, copy the rest of your application code.
COPY src/ src/
# DO NOT COPY THE .ENV FILE
# COPY .env . 
ENV PYTHONPATH /app
# Set environment variables for Streamlit and Python
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Expose the port that Streamlit runs on
EXPOSE 8501

# 4. The command to run when the container starts.
CMD ["streamlit", "run", "src/inventory_dashboard_streamlit.py"]
