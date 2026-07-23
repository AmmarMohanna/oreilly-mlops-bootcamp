import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from pathlib import Path
from data_pipeline.preprocessing import load_and_preprocess_data  


# Ensure the directory exists
os.makedirs(os.path.dirname('model/'), exist_ok=True)

def train():
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(Path.cwd() / 'mlflow.db').resolve()}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Income_Classification_Randomforest")
    with mlflow.start_run():

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", acc)

        for label, scores in report.items():
            if isinstance(scores, dict):  
                mlflow.log_metric(f"{label}_precision", scores["precision"])
                mlflow.log_metric(f"{label}_recall", scores["recall"])

        # Save model and preprocessor
        joblib.dump(clf, "model/rf_model.pkl")
        joblib.dump(preprocessor, "model/preprocessor.pkl")
        mlflow.sklearn.log_model(
            clf,
            name="model",
            registered_model_name="IncomeClassifier"
        )
        print(f"Accuracy: {acc:.4f}")
        print(f"MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    train()
