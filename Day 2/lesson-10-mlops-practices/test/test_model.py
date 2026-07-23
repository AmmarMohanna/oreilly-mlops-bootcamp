import pytest
import joblib
from sklearn.metrics import f1_score
from data_pipeline.preprocessing import load_and_preprocess_data 
from pathlib import Path

MODEL_PATH = Path("model/rf_model.pkl")

# Test Model Accuracy and F1 Score
@pytest.fixture(scope="module")
def setup():
    if not MODEL_PATH.exists():
        from train import train
        train()

    # Load and preprocess the data
    _, X_test, _, y_test, _ = load_and_preprocess_data()
    clf = joblib.load(MODEL_PATH)
    
    return clf, X_test, y_test

def test_f1_score(setup):
    clf, X_test, y_test = setup
    y_pred = clf.predict(X_test)  # Predict using preprocessed test data
    
    # Calculate the F1 score using weighted average
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Assert that the F1 score is above 0.80
    assert f1 > 0.80, f"F1 score is too low: {f1}"

def test_model_output(setup):
    clf, X_test, y_test = setup
    y_pred = clf.predict(X_test)  # Predict using preprocessed test data
    
    # Ensure predictions are of the correct shape
    assert y_pred.shape[0] == y_test.shape[0], "Mismatch in the number of predictions and true labels"
