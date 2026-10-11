import os
import joblib
import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_PATH = "tourism_project/deployment/best_model.pkl"


def main():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")

    cat_cols = Xtrain.select_dtypes(include=["object"]).columns.tolist()
    num_cols = Xtrain.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("tourism_package_prediction")

    with mlflow.start_run(run_name="random_forest_grid_search"):
        grid.fit(Xtrain, ytrain)

        best_model = grid.best_estimator_
        preds = best_model.predict(Xtest)

        metrics = {
            "accuracy": accuracy_score(ytest, preds),
            "precision": precision_score(ytest, preds),
            "recall": recall_score(ytest, preds),
            "f1": f1_score(ytest, preds),
        }

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)

        print("Best Parameters:", grid.best_params_)
        print("\nClassification Report:")
        print(classification_report(ytest, preds))

        os.makedirs("tourism_project/deployment", exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
