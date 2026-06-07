#!/bin/bash

# Quick deploy script for llama-server on Kubernetes / minikube.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="${IMAGE_NAME:-my-llama-server:v1}"

echo "llama-server K8s deployment"
echo "==========================="

echo
echo "Step 1: building Docker image: ${IMAGE_NAME}"
cd "$PROJECT_DIR"
docker build -t "$IMAGE_NAME" -f Dockerfile .

echo
echo "Step 2: checking Kubernetes cluster"
if ! kubectl cluster-info >/dev/null 2>&1; then
    echo "Kubernetes cluster not found. Start minikube or another cluster first."
    exit 1
fi
kubectl cluster-info

if command -v minikube >/dev/null 2>&1; then
    echo
    echo "Step 3: loading image into minikube"
    minikube image load "$IMAGE_NAME"
fi

echo
echo "Step 4: applying manifest"
kubectl apply -f k8s/manifest.yaml

echo
echo "Step 5: waiting for llama-server deployment"
kubectl wait --for=condition=available deployment/llama-server-deployment --timeout=300s

echo
echo "Deployment status:"
kubectl get deploy,svc,pods -l app=llama-server

echo
echo "Logs:"
echo "  kubectl logs -f deployment/llama-server-deployment"
echo
echo "Port forwarding localhost:8080 -> llama-server-service:8080"
echo "Press Ctrl+C to stop port-forward."

kubectl port-forward service/llama-server-service 8080:8080
