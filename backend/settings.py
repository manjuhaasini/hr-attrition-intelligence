import os
import streamlit as st

def get_settings():
    def read(key, fallback=""):
        try:
            value = st.secrets.get(key)
            if value is not None:
                return value
        except Exception:
            pass
        return os.getenv(key, fallback)
    return {
        "SUPABASE_URL": read("SUPABASE_URL"),
        "SUPABASE_PUBLISHABLE_KEY": read("SUPABASE_PUBLISHABLE_KEY") or read("SUPABASE_ANON_KEY"),
        "SUPABASE_SECRET_KEY": read("SUPABASE_SECRET_KEY") or read("SUPABASE_SERVICE_ROLE_KEY"),
        "MODEL_PATH": read("MODEL_PATH", "attrition_model.joblib"),
    }
