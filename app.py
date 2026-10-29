
from pathlib import Path
import traceback

import pandas as pd
import streamlit as st
import plotly.express as px

from backend.settings import get_settings
from backend.auth import sign_in, verify_admin, sign_out
from backend.database import (
    get_admin_client,
    list_records,
    save_employee,
    update_employee,
    import_historical_csv,
    get_prediction_history,
    get_audit_log,
)
from backend.prediction import load_bundle, predict_values


BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="HR Attrition Intelligence",
    page_icon="📊",
    layout="wide",
)
st.title("HR Attrition Intelligence")
# Display success messages saved before a page rerun.
if st.session_state.get("success_message"):
    st.success(st.session_state.pop("success_message"))
st.caption(
    "Admin-only employee records and attrition decision support. "
    "Predictions are estimates, not certainty."
)

cfg = get_settings()


@st.cache_resource
def load_model_bundle():
    return load_bundle(
        BASE / cfg.get("MODEL_PATH", "attrition_model.joblib")
    )


@st.cache_data
def load_reference():
    df = pd.read_csv(BASE / "AttritionData.csv")
    return df.drop(
        columns=[
            "Attrition",
            "EmployeeCount",
            "EmployeeNumber",
            "Over18",
            "StandardHours",
        ],
        errors="ignore",
    )


if "access_token" not in st.session_state:
    st.subheader("Admin sign in")

    with st.form("login_form"):
        email = st.text_input("Admin email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(
            "Sign in",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            session = sign_in(email.strip(), password)
            st.session_state.update({
                "access_token": session["access_token"],
                "user_id": session["user_id"],
                "email": session["email"],
            })
            st.rerun()
        except Exception as exc:
            st.error(f"Sign-in failed: {exc}")

    st.info(
        "Admin accounts must be created in Supabase Authentication "
        "and approved in the profiles table."
    )
    st.stop()


try:
    if not verify_admin(
        st.session_state["access_token"],
        st.session_state["user_id"],
    ):
        st.error("This account is not an active admin.")
        if st.button("Clear session"):
            st.session_state.clear()
            st.rerun()
        st.stop()
except Exception as exc:
    st.error(f"Could not verify admin access: {exc}")
    if st.button("Return to login"):
        st.session_state.clear()
        st.rerun()
    st.stop()


db = get_admin_client()
bundle = load_model_bundle()
features = bundle["feature_columns"]
reference = load_reference()


with st.sidebar:
    st.write("Signed in as")
    st.caption(st.session_state.get("email", "Admin"))

    page = st.radio(
        "Navigation",
        [
            "Overview Dashboard",
            "Employee Records",
            "Add Employee",
            "Update Employee",
            "Prediction History",
            "Audit Log",
            "Dataset Import",
        ],
    )

    if st.button("Sign out", use_container_width=True):
        sign_out()
        st.session_state.clear()
        st.rerun()


def read_records():
    raw = list_records(db)

    if not raw:
        return pd.DataFrame()

    frame = pd.DataFrame(raw)
    expanded = pd.json_normalize(
        frame["employee_data"].apply(
            lambda value: value if isinstance(value, dict) else {}
        )
    )

    return pd.concat(
        [frame.drop(columns=["employee_data"]), expanded],
        axis=1,
    )


def employee_form(defaults, key):
    values = {}
    categorical = reference.select_dtypes(
        include="object"
    ).columns.tolist()

    groups = {
        "Personal and role details": [
            "Age", "Gender", "Department", "JobRole",
            "Education", "EducationField", "MaritalStatus",
        ],
        "Work and experience": [
            "BusinessTravel", "DistanceFromHome",
            "NumCompaniesWorked", "TotalWorkingYears",
            "YearsAtCompany", "YearsInCurrentRole",
            "YearsSinceLastPromotion", "YearsWithCurrManager",
        ],
        "Satisfaction and engagement": [
            "EnvironmentSatisfaction", "JobInvolvement",
            "JobSatisfaction", "RelationshipSatisfaction",
            "WorkLifeBalance", "OverTime",
        ],
        "Compensation and other factors": [
            "DailyRate", "HourlyRate", "MonthlyIncome",
            "MonthlyRate", "PercentSalaryHike",
            "PerformanceRating", "StockOptionLevel",
            "TrainingTimesLastYear", "JobLevel",
        ],
    }

    with st.form(key):
        st.caption(
            "Use the same field definitions and units as the original dataset."
        )

        for group, cols in groups.items():
            cols = [col for col in cols if col in features]

            if not cols:
                continue

            with st.expander(
                group,
                expanded=(group == "Personal and role details"),
            ):
                left, right = st.columns(2)

                for i, col in enumerate(cols):
                    target = left if i % 2 == 0 else right
                    default = defaults.get(col)

                    if col in categorical:
                        options = sorted(
                            reference[col].dropna().astype(str).unique().tolist()
                        )

                        if default is None or str(default) not in options:
                            mode = str(reference[col].mode().iloc[0])
                            default = mode if mode in options else options[0]

                        values[col] = target.selectbox(
                            col,
                            options,
                            index=options.index(str(default)),
                            key=f"{key}_{col}",
                        )
                    else:
                        series = reference[col]
                        lo = int(series.min())
                        hi = int(series.max())

                        default = (
                            int(round(series.median()))
                            if default is None
                            else int(float(default))
                        )

                        values[col] = target.number_input(
                            col,
                            min_value=lo,
                            max_value=hi,
                            value=max(lo, min(hi, default)),
                            step=1,
                            key=f"{key}_{col}",
                        )

        submitted = st.form_submit_button(
            "Save and calculate prediction",
            type="primary",
            use_container_width=True,
        )

    return values, submitted


df = read_records()


if page == "Overview Dashboard":
    st.header("Overview Dashboard")

    if df.empty:
        st.info("No records yet. Use Dataset Import to import the historical CSV.")
        st.stop()

    actual = df.get("actual_attrition", pd.Series(dtype=object))
    pred = df.get("predicted_attrition", pd.Series(dtype=object))
    prob = df.get("attrition_probability", pd.Series(dtype=float))

    a, b, c, d = st.columns(4)
    a.metric("Employee records", f"{len(df):,}")
    b.metric("Known actual outcomes", f"{actual.notna().sum():,}")
    c.metric("Predicted attrition: Yes", f"{(pred == 'Yes').sum():,}")
    d.metric(
        "Estimated probability ≥ 50%",
        f"{(prob.fillna(0) >= 0.5).sum():,}",
    )

    left, right = st.columns(2)

    with left:
        if actual.notna().any():
            chart_data = (
                actual.fillna("Unknown")
                .value_counts()
                .rename_axis("Actual attrition")
                .reset_index(name="Employees")
            )
            st.plotly_chart(
                px.bar(
                    chart_data,
                    x="Actual attrition",
                    y="Employees",
                    title="Actual Attrition Outcomes",
                ),
                use_container_width=True,
            )

        if "Department" in df:
            chart_data = (
                df.assign(
                    Department=df["Department"].fillna("Unknown").astype(str)
                )
                .groupby("Department")["attrition_probability"]
                .mean()
                .reset_index()
            )
            st.plotly_chart(
                px.bar(
                    chart_data,
                    x="Department",
                    y="attrition_probability",
                    title="Mean Estimated Probability by Department",
                ),
                use_container_width=True,
            )

    with right:
        if pred.notna().any():
            chart_data = (
                pred.fillna("Not scored")
                .value_counts()
                .rename_axis("Prediction")
                .reset_index(name="Employees")
            )
            st.plotly_chart(
                px.pie(
                    chart_data,
                    names="Prediction",
                    values="Employees",
                    title="Predicted Attrition",
                ),
                use_container_width=True,
            )

        if "JobLevel" in df:
            chart_data = (
                df.assign(JobLevel=df["JobLevel"].fillna("Unknown").astype(str))
                .groupby("JobLevel")["attrition_probability"]
                .mean()
                .reset_index()
            )
            st.plotly_chart(
                px.bar(
                    chart_data,
                    x="JobLevel",
                    y="attrition_probability",
                    title="Mean Estimated Probability by Job Level",
                ),
                use_container_width=True,
            )

    options = [
        col for col in [
            "Department", "JobRole", "JobLevel", "OverTime",
            "BusinessTravel", "MaritalStatus", "EducationField", "Gender",
        ]
        if col in df
    ]

    if options:
        group = st.selectbox("Analyse by feature", options)

        chart_data = (
            df.assign(_group=df[group].fillna("Unknown").astype(str))
            .groupby("_group")
            .agg(
                Employees=("record_key", "count"),
                Mean_probability=("attrition_probability", "mean"),
            )
            .reset_index()
            .rename(columns={"_group": group})
        )

        st.dataframe(chart_data, use_container_width=True, hide_index=True)


elif page == "Employee Records":
    st.header("Employee Records")

    if df.empty:
        st.info("Import the historical CSV first.")
        st.stop()

    filters = [
        col for col in [
            "source_type", "Department", "JobRole", "JobLevel",
            "actual_attrition", "predicted_attrition",
        ]
        if col in df
    ]

    filtered = df.copy()

    with st.expander("Filters", expanded=True):
        cols = st.columns(3)

        for i, col in enumerate(filters):
            options = sorted(
                filtered[col].fillna("Unknown").astype(str).unique().tolist()
            )
            chosen = cols[i % 3].multiselect(col, options, key=f"filter_{col}")

            if chosen:
                filtered = filtered[
                    filtered[col].fillna("Unknown").astype(str).isin(chosen)
                ]

    show = [
        col for col in [
            "record_key", "source_type", "EmployeeNumber",
            "Department", "JobRole", "JobLevel", "actual_attrition",
            "predicted_attrition", "attrition_probability", "updated_at",
        ]
        if col in filtered
    ]

    st.dataframe(filtered[show], use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered CSV",
        filtered[show].to_csv(index=False).encode(),
        "employee_records.csv",
        "text/csv",
    )


elif page == "Add Employee":
    st.header("Add Employee")
    values, submitted = employee_form({}, "add_employee")

    if submitted:
        try:
            pred, prob = predict_values(bundle, values)
            save_employee(
                db,
                values,
                pred,
                prob,
                st.session_state["user_id"],
            )
            st.session_state["success_message"] = (
                f"Employee saved successfully. Prediction: {pred}; "
                f"estimated attrition probability: {prob:.1%}."
            )
            st.rerun()
        except Exception:
            st.error("Could not save the employee record.")
            st.code(traceback.format_exc())


elif page == "Update Employee":
    st.header("Update Employee")

    if df.empty:
        st.info("Import the CSV or add a record first.")
        st.stop()

    keys = df["record_key"].astype(str).tolist()
    chosen = st.selectbox("Select employee record", keys)

    if st.button("Load selected record"):
        st.session_state["editing_key"] = chosen

    if st.session_state.get("editing_key"):
        record = next(
            (
                row for row in list_records(db)
                if row["record_key"] == st.session_state["editing_key"]
            ),
            None,
        )

        if not record:
            st.error("Record not found.")
            st.stop()

        values, submitted = employee_form(
            record.get("employee_data") or {},
            "update_employee",
        )

        options = ["Unknown", "Yes", "No"]
        old_actual = record.get("actual_attrition") or "Unknown"

        actual = st.selectbox(
            "Actual attrition (only if known)",
            options,
            index=options.index(old_actual) if old_actual in options else 0,
        )

        if submitted:
            try:
                pred, prob = predict_values(bundle, values)
                update_employee(
                    db,
                    record["id"],
                    values,
                    pred,
                    prob,
                    None if actual == "Unknown" else actual,
                    st.session_state["user_id"],
                )
                st.session_state["success_message"] = (
                    f"Employee updated successfully. Prediction: {pred}; "
                    f"estimated attrition probability: {prob:.1%}."
                )
                st.rerun()
            except Exception:
                st.error("Could not update the employee record.")
                st.code(traceback.format_exc())


elif page == "Prediction History":
    st.header("Prediction History")
    rows = get_prediction_history(db)

    if rows:
        history = pd.DataFrame(rows)
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button(
            "Download history",
            history.to_csv(index=False).encode(),
            "prediction_history.csv",
            "text/csv",
        )
    else:
        st.info("No stored predictions yet.")


elif page == "Audit Log":
    st.header("Audit Log")
    rows = get_audit_log(db)

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No audit entries yet.")


elif page == "Dataset Import":
    st.header("Import historical dataset")

    st.warning(
        "The original CSV remains unchanged. This stores a copy in Supabase "
        "and scores each row using your saved model."
    )

    uploaded = st.file_uploader(
        "Upload AttritionData.csv",
        type="csv",
    )

    if uploaded:
        try:
            preview = pd.read_csv(uploaded)
            st.write(f"{len(preview):,} rows × {len(preview.columns)} columns")
            st.dataframe(preview.head(8), use_container_width=True)

            if st.button(
                "Import records and calculate predictions",
                type="primary",
            ):
                try:
                    result = import_historical_csv(
                        db,
                        preview,
                        bundle,
                        st.session_state["user_id"],
                    )

                    # Support either an integer count or a dictionary result.
                    if isinstance(result, int):
                        import_message = f"Imported or updated {result:,} records successfully."
                    elif isinstance(result, dict):
                        processed = result.get("processed", result.get("count", 0))
                        scored = result.get("scored", processed)
                        import_message = (
                            f"Import completed. Processed {processed:,} records; "
                            f"calculated {scored:,} predictions."
                        )
                    else:
                        import_message = f"Import completed successfully. Result: {result}"

                    st.session_state["success_message"] = import_message
                    st.rerun()

                except Exception:
                    st.error("Import failed. Full traceback:")
                    st.code(traceback.format_exc())

        except Exception:
            st.error("Could not read the uploaded CSV.")
            st.code(traceback.format_exc())


st.divider()
st.caption(
    "Predictions are statistical estimates and must not be used "
    "as the sole basis for employment decisions."
)
