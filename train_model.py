
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parent


def train_and_save():
    data_path = BASE / "AttritionData.csv"
    model_path = BASE / "attrition_model.joblib"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    if "Attrition" not in df.columns:
        raise ValueError("Dataset must contain an Attrition column.")

    # Keep the original CSV unchanged.
    df = df.drop(
        columns=[
            "EmployeeCount",
            "EmployeeNumber",
            "Over18",
            "StandardHours",
        ],
        errors="ignore",
    )

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError(
            "Unexpected values found in the Attrition column."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_features = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=1000,
                ),
            ),
        ]
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"model__C": [0.01, 0.1, 1, 10]},
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    print("Training model. Please wait...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, test_probabilities)

    bundle = {
        "model": best_model,
        "feature_columns": X.columns.tolist(),
        "best_params": search.best_params_,
        "cv_roc_auc": float(search.best_score_),
    }

    # Save the newly trained model bundle.
    joblib.dump(bundle, model_path)

    print("\nTraining completed.")
    print(f"Model saved to: {model_path}")
    print(f"scikit-learn version used: compatible with this environment")
    print("Best parameters:", search.best_params_)
    print(f"Mean CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Features: {len(X.columns)}")


if __name__ == "__main__":
    train_and_save()
