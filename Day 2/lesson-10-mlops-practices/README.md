# Lesson 10: End-to-End MLOps Practices

This demo trains an Adult Income classifier, logs it to a local MLflow store,
tests it, serves predictions and Prometheus metrics, and deploys the service to
Docker Desktop Kubernetes.

## Set up the lesson environment

From the repository root:

```bash
conda activate lesson-10-mlops-practices
cd "Day 2/lesson-10-mlops-practices"
python -m pip install -r requirements.txt
python -m pip check
```

## Train and test

No separate MLflow server is required for training:

```bash
python train.py
python -m pytest -q
```

Training creates ignored `model/`, `mlruns/`, and `mlflow.db` artifacts. To
inspect the recorded run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Open `http://localhost:5001`.

## Run the API locally

```bash
python app.py
```

In another terminal:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
  }'

curl http://localhost:8000/metrics
```

Stop the API with `Ctrl+C`.

## Run with Docker

Train first so the `model/` directory exists, then:

```bash
docker build -f dockerfile -t income-flask-app:latest .
docker run --rm -p 8000:8000 income-flask-app:latest
```

Test `/health`, `/predict`, and `/metrics` with the commands above.

## Run on Kubernetes

Docker Desktop Kubernetes can use the locally built image:

```bash
docker build -f dockerfile -t income-flask-app:latest .
kubectl apply -f k8s-deployment.yaml
kubectl rollout status deployment/income-flask-app --timeout=120s
curl http://localhost:30800/health
```

Deploy the optional monitoring stack:

```bash
kubectl apply -f monitoring-deployment.yaml
kubectl rollout status deployment/prometheus --timeout=120s
kubectl rollout status deployment/grafana --timeout=120s
```

Services:

- Flask API: `http://localhost:30800`
- Prometheus: `http://localhost:30909`
- Grafana: `http://localhost:30009` (`admin` / `admin`)

In Grafana, add `http://prometheus-service:9090` as the Prometheus data source.
Useful metrics include `predict_requests_total`,
`predict_exceptions_total`, and `predict_request_latency_seconds`.

## Clean up

```bash
kubectl delete -f monitoring-deployment.yaml --ignore-not-found
kubectl delete -f k8s-deployment.yaml --ignore-not-found
```
