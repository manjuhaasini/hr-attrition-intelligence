
from pathlib import Path

import joblib
import pandas as pd


def load_bundle(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {path.resolve()}"
        )

    bundle = joblib.load(path)

    if (
        not isinstance(bundle, dict)
        or "model" not in bundle
        or "feature_columns" not in bundle
    ):
        raise ValueError(
            "Expected model bundle keys 'model' and 'feature_columns'."
        )

    return bundle


def predict_values(bundle, payload):
    features = bundle["feature_columns"]

    missing = [
        col for col in features
        if col not in payload or payload[col] is None
    ]

    if missing:
        raise ValueError(
            "Missing required fields: " + ", ".join(missing)
        )

    row = pd.DataFrame(
        [{col: payload[col] for col in features}],
        columns=features,
    )

    model = bundle["model"]

    # Handle a missing multi_class attribute in an older
    # serialized LogisticRegression estimator.
    estimator = (
        model.named_steps.get("model", model)
        if hasattr(model, "named_steps")
        else model
    )

    if (
        hasattr(estimator, "solver")
        and not hasattr(estimator, "multi_class")
    ):
        estimator.multi_class = "auto"

    try:
        prediction = model.predict(row)
        probabilities = model.predict_proba(row)

        pred = int(prediction[0])
        prob = float(probabilities[0][1])

    except Exception as exc:
        raise RuntimeError(
            f"Model prediction failed: {type(exc).__name__}: {exc}"
        ) from exc

    return ("Yes" if pred == 1 else "No"), prob
