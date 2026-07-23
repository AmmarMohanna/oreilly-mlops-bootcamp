# Day 2: Applied MLOps

Day 2 contains four independent demos. Use a separate Python 3.11 Conda
environment whose name exactly matches the lesson folder.

## Prerequisites

- Conda
- Docker Desktop
- Kubernetes enabled in Docker Desktop
- `kubectl`

Confirm the local services before starting:

```bash
docker info
kubectl cluster-info
kubectl top nodes
```

## One-time environment setup

Run these commands from the repository root:

```bash
conda create -n lesson-2-kubernetes-basics python=3.11 -y
conda run -n lesson-2-kubernetes-basics \
  python -m pip install -r "Day 2/lesson-2-kubernetes-basics/age_detection/requirements.txt"

conda create -n lesson-5-kubernetes-hpa python=3.11 -y
conda run -n lesson-5-kubernetes-hpa \
  python -m pip install -r "Day 2/lesson-5-kubernetes-hpa/age_detection/requirements.txt"

conda create -n lesson-7-ab-testing python=3.11 -y
conda run -n lesson-7-ab-testing \
  python -m pip install -r "Day 2/lesson-7-ab-testing/requirements.txt"

conda create -n lesson-10-mlops-practices python=3.11 -y
conda run -n lesson-10-mlops-practices \
  python -m pip install -r "Day 2/lesson-10-mlops-practices/requirements.txt"
```

If an environment already exists, skip its `conda create` command and run only
the matching install command.

## Demo guides

1. [Kubernetes basics](lesson-2-kubernetes-basics/README.md)
2. [Kubernetes horizontal pod autoscaling](lesson-5-kubernetes-hpa/README.md)
3. [A/B testing](lesson-7-ab-testing/README.md)
4. [End-to-end MLOps practices](lesson-10-mlops-practices/README.md)

The two age-detection lessons intentionally reuse the `age-detect` Kubernetes
resource names and NodePort `30600`. Clean up one lesson before starting the
other.
