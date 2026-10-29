"""Small login / register / history module for the dashboard (prototype-level security).
Passwords are stored as salted PBKDF2-SHA256 hashes in users.json next to this file."""
import os, re, json, hmac, hashlib, secrets, datetime
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
USERS = os.path.join(HERE, "users.json")
ITER = 120_000


def _load():
    try:
        with open(USERS, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(d):
    with open(USERS, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)


def _hash(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), ITER).hex()


def register(name, pw):
    name = name.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.@-]{3,40}", name):
        return "Username: 3-40 characters (letters, digits, . _ - @)."
    if len(pw) < 6:
        return "Password must have at least 6 characters."
    d = _load()
    if name in d:
        return "That username already exists."
    salt = secrets.token_hex(16)
    d[name] = dict(salt=salt, hash=_hash(pw, salt), created=str(datetime.date.today()), history=[])
    _save(d)
    return None


def check(name, pw):
    u = _load().get(name.strip().lower())
    return bool(u) and hmac.compare_digest(u["hash"], _hash(pw, u["salt"]))


def add_history(name, record):
    d = _load()
    if name in d:
        d[name]["history"].append(record)
        _save(d)


def get_history(name):
    return _load().get(name, {}).get("history", [])


def gate():
    """Show the login/register screen until the user signs in. Returns the username."""
    if st.session_state.get("user"):
        st.sidebar.write(f"Signed in as **{st.session_state['user']}**")
        if st.sidebar.button("Log out"):
            st.session_state.clear()
            st.rerun()
        return st.session_state["user"]
    st.title("🎯 Career Skill Gap Analyzer")
    st.caption("Sign in to analyse your resume and keep a history of your readiness scores.")
    t1, t2 = st.tabs(["Log in", "Create account"])
    with t1:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Log in"):
                if check(u, p):
                    st.session_state["user"] = u.strip().lower()
                    st.rerun()
                else:
                    st.error("Wrong username or password.")
    with t2:
        with st.form("register"):
            u = st.text_input("Choose a username")
            p = st.text_input("Choose a password (min 6 characters)", type="password")
            p2 = st.text_input("Repeat password", type="password")
            if st.form_submit_button("Create account"):
                err = "Passwords do not match." if p != p2 else register(u, p)
                if err:
                    st.error(err)
                else:
                    st.session_state["user"] = u.strip().lower()
                    st.rerun()
    st.stop()
