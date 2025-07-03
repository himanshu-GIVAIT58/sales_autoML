# Use an official Python image as the base image
FROM python:3.11-slim

# Set the working directory for all subsequent commands
WORKDIR /app

# 1. Copy only the requirements file first to leverage Docker layer caching.
#    This step is now separate and placed before copying all other code.
COPY src/requirements.txt src/requirements.txt

# 2. Install Python dependencies.
#    This layer will only be re-built if the requirements.txt file changes.
RUN pip install --no-cache-dir -r src/requirements.txt

# 3. Now, copy the rest of your application code and environment file.
#    This maintains your desired /app/src/... structure.
COPY src/ src/
COPY .env .
ENV PYTHONPATH /app
# Set environment variables for Streamlit and Python
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Expose the port that Streamlit runs on
EXPOSE 8501

# 4. The command to run when the container starts.
#    This path is correct for your specified structure.
CMD ["streamlit", "run", "src/inventory_dashboard_streamlit.py"]
