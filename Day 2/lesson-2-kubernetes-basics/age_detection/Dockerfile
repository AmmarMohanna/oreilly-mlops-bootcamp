# Match the Python version used by the lesson Conda environment
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
