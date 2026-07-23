# Lesson 7: A/B Testing

This demo trains logistic-regression and random-forest diabetes classifiers,
alternates requests between them, and reports performance on labeled samples.

## Set up the lesson environment

From the repository root:

```bash
conda activate lesson-7-ab-testing
cd "Day 2/lesson-7-ab-testing"
python -m pip install -r requirements.txt
python -m pip check
```

## Train the models

```bash
python training.py
```

This creates the ignored `models/` directory.

## Start the API

```bash
python app.py
```

The API runs at `http://localhost:8000`. Confirm it is ready:

```bash
curl http://localhost:8000/health
```

## Send one prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 50,
      "gender": "Male",
      "smoking_history": "never",
      "hypertension": 0,
      "heart_disease": 0,
      "bmi": 25.0,
      "HbA1c_level": 6.0,
      "blood_glucose_level": 140
    },
    "true_label": 1
  }'
```

## Run the A/B simulation

Keep the API running. In another terminal, activate the same environment and
return to the lesson directory:

```bash
conda activate lesson-7-ab-testing
cd "Day 2/lesson-7-ab-testing"
python simulator.py
python performance.py
```

The simulator sends exactly 100 labeled rows sampled from the dataset. The
performance command prints request count and accuracy for models A and B.

Stop the API with `Ctrl+C`.
