from supabase import create_client
from backend.settings import get_settings

def sign_in(email, password):
    cfg = get_settings()
    client = create_client(cfg["SUPABASE_URL"], cfg["SUPABASE_PUBLISHABLE_KEY"])
    result = client.auth.sign_in_with_password({"email": email, "password": password})
    if not result.session or not result.user:
        raise RuntimeError("Supabase did not return a valid session.")
    return {"access_token": result.session.access_token, "user_id": result.user.id, "email": result.user.email}

def verify_admin(access_token, user_id):
    cfg = get_settings()
    auth_client = create_client(cfg["SUPABASE_URL"], cfg["SUPABASE_PUBLISHABLE_KEY"])
    result = auth_client.auth.get_user(access_token)
    if not result.user or result.user.id != user_id:
        return False
    from backend.database import get_admin_client
    db = get_admin_client()
    profile = db.table("profiles").select("role,is_active").eq("id", user_id).maybe_single().execute().data
    return bool(profile and profile.get("role") == "admin" and profile.get("is_active") is True)

def sign_out():
    return None
