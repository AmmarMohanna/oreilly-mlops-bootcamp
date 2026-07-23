# Lesson 5: Kubernetes Horizontal Pod Autoscaling

This demo deploys the age-detection API with CPU requests and limits, then uses
Kubernetes HPA to scale it under load.

## Set up the lesson environment

From the repository root:

```bash
conda activate lesson-5-kubernetes-hpa
cd "Day 2/lesson-5-kubernetes-hpa"
python -m pip install -r age_detection/requirements.txt
python -m pip check
```

## Build and deploy

```bash
docker build -t age-detect:latest ./age_detection
kubectl apply -f k8s-deployment.yml
kubectl rollout status deployment/age-detect --timeout=120s
curl http://localhost:30600/health
```

For Minikube, run `minikube image load age-detect:latest` before applying the
manifest.

## Confirm metrics are available

```bash
kubectl top nodes
kubectl top pods
```

Docker Desktop normally includes metrics-server. On Minikube, enable it if the
commands above fail:

```bash
minikube addons enable metrics-server
```

## Apply and observe the HPA

```bash
kubectl apply -f hpa.yml
kubectl get hpa
kubectl describe hpa age-detect-hpa
```

Start a continuous synthetic-inference load:

```bash
kubectl run loadgen --restart=Never --image=busybox:1.36 -- \
  /bin/sh -c 'while true; do
    wget -q -O- http://age-detect:5000/stress >/dev/null
    sleep 0.02
  done'
```

Watch scaling in separate terminals:

```bash
kubectl get hpa -w
```

```bash
kubectl get pods -w
```

HPA uses a moving metrics window, so a scale-up can take one or two minutes.

## Clean up

```bash
kubectl delete pod loadgen --ignore-not-found
kubectl delete -f hpa.yml
kubectl delete -f k8s-deployment.yml
```
