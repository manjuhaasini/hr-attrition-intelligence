
import uuid
from datetime import datetime, timezone

import pandas as pd
from supabase import create_client

from backend.settings import get_settings


def get_admin_client():
    cfg = get_settings()

    if not cfg.get("SUPABASE_URL") or not cfg.get("SUPABASE_SECRET_KEY"):
        raise RuntimeError(
            "Configure SUPABASE_URL and SUPABASE_SECRET_KEY "
            "in Streamlit secrets."
        )

    return create_client(
        cfg["SUPABASE_URL"],
        cfg["SUPABASE_SECRET_KEY"],
    )


def now():
    return datetime.now(timezone.utc).isoformat()



def list_records(client, limit=10000):
    records = []
    page_size = 500
    start = 0

    while start < limit:
        end = min(start + page_size, limit) - 1

        response = (
            client.table("employee_records")
            .select("*")
            .order("updated_at", desc=True)
            .range(start, end)
            .execute()
        )

        batch = response.data or []
        records.extend(batch)

        if len(batch) < (end - start + 1):
            break

        start += len(batch)

    return records



def get_record(client, record_id):
    response = (
        client.table("employee_records")
        .select("*")
        .eq("id", record_id)
        .maybe_single()
        .execute()
    )
    return response.data


def audit(client, record_id, actor_id, action, details):
    client.table("employee_audit_log").insert({
        "record_id": record_id,
        "actor_id": actor_id,
        "action": action,
        "details": details,
        "created_at": now(),
    }).execute()


def save_prediction(
    client,
    record_id,
    predicted,
    probability,
    actor_id,
):
    client.table("prediction_history").insert({
        "record_id": record_id,
        "predicted_attrition": predicted,
        "probability_yes": float(probability),
        "model_version": "lasso-logistic-bundle",
        "predicted_at": now(),
        "created_by": actor_id,
    }).execute()


def save_employee(
    client,
    payload,
    predicted,
    probability,
    actor_id,
    source_type="live",
    actual_attrition=None,
    record_key=None,
):
    key = record_key or f"live:{uuid.uuid4()}"

    employee = {
        "record_key": key,
        "source_type": source_type,
        "employee_data": payload,
        "actual_attrition": actual_attrition,
        "predicted_attrition": predicted,
        "attrition_probability": float(probability),
        "created_by": actor_id,
        "updated_by": actor_id,
        "updated_at": now(),
    }

    response = (
        client.table("employee_records")
        .upsert(employee, on_conflict="record_key")
        .execute()
    )

    if response.data:
        saved = response.data[0]
    else:
        saved = (
            client.table("employee_records")
            .select("*")
            .eq("record_key", key)
            .single()
            .execute()
            .data
        )

    save_prediction(
        client,
        saved["id"],
        predicted,
        probability,
        actor_id,
    )

    audit(
        client,
        saved["id"],
        actor_id,
        "create" if source_type == "live" else "historical_import",
        {"source_type": source_type},
    )

    return saved


def update_employee(
    client,
    record_id,
    payload,
    predicted,
    probability,
    actual_attrition,
    actor_id,
):
    old = get_record(client, record_id)

    if not old:
        raise ValueError("Record not found.")

    (
        client.table("employee_records")
        .update({
            "employee_data": payload,
            "actual_attrition": actual_attrition,
            "predicted_attrition": predicted,
            "attrition_probability": float(probability),
            "updated_by": actor_id,
            "updated_at": now(),
        })
        .eq("id", record_id)
        .execute()
    )

    save_prediction(
        client,
        record_id,
        predicted,
        probability,
        actor_id,
    )

    audit(
        client,
        record_id,
        actor_id,
        "update",
        {
            "previous_prediction": old.get("predicted_attrition"),
            "new_prediction": predicted,
        },
    )


def import_historical_csv(client, frame, bundle, actor_id):
    if "Attrition" not in frame.columns:
        raise ValueError("CSV must include the Attrition column.")

    features = bundle["feature_columns"]
    model = bundle["model"]

    missing_columns = [
        col for col in features
        if col not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSV is missing model features: "
            + ", ".join(missing_columns)
        )

    # Compatibility handling for an older serialized
    # LogisticRegression estimator.
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

    batch = []
    imported = 0

    for idx, series in frame.iterrows():
        payload = {}

        for col in features:
            value = series[col]

            if pd.isna(value):
                payload[col] = None
            elif hasattr(value, "item"):
                payload[col] = value.item()
            else:
                payload[col] = value

        row_df = pd.DataFrame(
            [{col: payload[col] for col in features}],
            columns=features,
        )

        try:
            prediction = model.predict(row_df)
            probabilities = model.predict_proba(row_df)

            pred = int(prediction[0])
            prob = float(probabilities[0][1])

        except Exception as exc:
            raise RuntimeError(
                f"Model prediction failed for CSV row {idx + 1}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        employee_number = series.get("EmployeeNumber", idx + 1)

        record = {
            "record_key": f"historical:{employee_number}",
            "source_type": "historical",
            "employee_data": payload,
            "actual_attrition": str(series["Attrition"]).strip(),
            "predicted_attrition": "Yes" if pred == 1 else "No",
            "attrition_probability": prob,
            "created_by": actor_id,
            "updated_by": actor_id,
            "updated_at": now(),
        }

        batch.append(record)

        if len(batch) >= 100:
            (
                client.table("employee_records")
                .upsert(batch, on_conflict="record_key")
                .execute()
            )
            imported += len(batch)
            batch = []

    if batch:
        (
            client.table("employee_records")
            .upsert(batch, on_conflict="record_key")
            .execute()
        )
        imported += len(batch)

    return imported


def get_prediction_history(client, limit=5000):
    response = (
        client.table("prediction_history")
        .select("*")
        .order("predicted_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


def get_audit_log(client, limit=1000):
    response = (
        client.table("employee_audit_log")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []
