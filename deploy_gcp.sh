#!/bin/bash

# This script deploys infrastructure and application code changes to GCP.
# It does NOT migrate or change the database data.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Replace with your actual GCP Project ID.
GCP_PROJECT_ID="giva-dev-26385"
IMAGE_NAME="us-central1-docker.pkg.dev/${GCP_PROJECT_ID}/my-app-repo/sales-automl:v1"

echo "🚀 Starting deployment to GCP..."

# --- 1. Deploy Infrastructure Changes ---
echo "Applying Kubernetes configurations from the 'k8s/' directory..."
kubectl apply -f k8s/

# --- 2. Deploy Application Code ---
echo "📦 Building and pushing the latest application image..."
docker build -t "${IMAGE_NAME}" .
docker push "${IMAGE_NAME}"

echo "🚢 Restarting Kubernetes deployment to pull the new image..."
kubectl rollout restart deployment/sales-automl-app-deployment

echo "🎉 Deployment complete!"
echo "Run 'kubectl get pods -w' to monitor the application rollout."
