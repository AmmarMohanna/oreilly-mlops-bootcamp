# Lesson 2: Kubernetes Basics

This demo runs an OpenCV age-detection Flask API locally, in Docker, and on a
local Kubernetes cluster.

## Set up the lesson environment

From the repository root:

```bash
conda activate lesson-2-kubernetes-basics
cd "Day 2/lesson-2-kubernetes-basics"
python -m pip install -r age_detection/requirements.txt
python -m pip check
```

## Run locally

```bash
python age_detection/app.py
```

In another terminal:

```bash
curl http://localhost:8080/health
curl -F "image=@/absolute/path/to/photo.jpg" \
  http://localhost:8080/detect_age
```

Stop the server with `Ctrl+C`.

## Run with Docker

The Docker build context must be the `age_detection` directory:

```bash
docker build -t age-detect:latest ./age_detection
docker run --rm -p 8080:8080 age-detect:latest
```

Test it with the same `/health` and `/detect_age` commands above.

## Run on Kubernetes

Docker Desktop Kubernetes can use the locally built image directly:

```bash
docker build -t age-detect:latest ./age_detection
kubectl apply -f k8s-deployment.yml
kubectl rollout status deployment/age-detect --timeout=120s
kubectl get pods,service
curl http://localhost:30600/health
curl -F "image=@/absolute/path/to/photo.jpg" \
  http://localhost:30600/detect_age
```

For Minikube, load the image before applying the manifest:

```bash
minikube image load age-detect:latest
```

## Clean up

```bash
kubectl delete -f k8s-deployment.yml
```
