# Use an official Python image as the base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Create necessary directories
RUN mkdir -p src/eda src/autogluon_models

# Copy the requirements file into the container
COPY src/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and models into the container
COPY src/ src/
COPY .env .

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose the port for Streamlit (default is 8501)
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "inventory_dashboard_streamlit.py"]
