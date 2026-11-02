"""Local SQLite authentication store for PPE Safety Monitor.

Security notes
--------------
* Passwords are NEVER stored in plain text. Each password is hashed with
  ``hashlib.pbkdf2_hmac('sha256', ...)`` using a per-user random salt
  (``secrets.token_hex(16)``) and 390,000 iterations (OWASP 2023 guidance).
* Stored format: ``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>``.
* Verification uses ``hmac.compare_digest`` to resist timing attacks.
* The database file ``ppe_app.db`` lives next to the executable when frozen
  with PyInstaller (so data persists between runs), or in the project root
  when running from source. It is deliberately NOT written into the
  read-only ``sys._MEIPASS`` temp folder.
"""
import hashlib
import hmac
import os
import secrets
import sqlite3

from src.config import get_asset_path

DB_NAME = "ppe_app.db"
PBKDF2_ITERATIONS = 390_000
HASH_ALGO = "sha256"
SALT_BYTES = 16

ROLE_ADMIN = "admin"
ROLE_USER = "user"

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"


# ---------------------------------------------------------------------------
# Path & connection helpers
# ---------------------------------------------------------------------------
def db_path():
    """Absolute path of ppe_app.db (next to the .exe when frozen)."""
    return get_asset_path(DB_NAME)


def _connect():
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def hash_password(password, salt_hex=None, iterations=PBKDF2_ITERATIONS):
    """Return 'pbkdf2_sha256$iters$salt_hex$hash_hex' for *password*."""
    if salt_hex is None:
        salt_hex = secrets.token_hex(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac(HASH_ALGO, password.encode("utf-8"),
                             bytes.fromhex(salt_hex), iterations)
    return f"pbkdf2_{HASH_ALGO}${iterations}${salt_hex}${dk.hex()}"


def verify_password(password, stored):
    """Timing-safe check of *password* against a stored hash string."""
    try:
        scheme, iters, salt_hex, hash_hex = stored.split("$")
        if scheme != f"pbkdf2_{HASH_ALGO}":
            return False
        dk = hashlib.pbkdf2_hmac(HASH_ALGO, password.encode("utf-8"),
                                 bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Schema / first-run seeding
# ---------------------------------------------------------------------------
def init_db():
    """Create the users table if needed.

    First-run logic: if the table ends up completely empty, seed a default
    Admin account (admin / admin123) and return True so callers can warn
    the user to change that password.
    """
    created_default = False
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'user'
            )
            """
        )
        cur = conn.execute("SELECT COUNT(*) AS n FROM users")
        if cur.fetchone()["n"] == 0:
            create_user(DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD,
                        ROLE_ADMIN, conn=conn)
            created_default = True
    return created_default


# ---------------------------------------------------------------------------
# CRUD API
# ---------------------------------------------------------------------------
def create_user(username, password, role=ROLE_USER, conn=None):
    """Insert a new user. Returns True on success, False on duplicate."""
    username = (username or "").strip()
    if not username or not password:
        raise ValueError("Username and password are required.")
    if role not in (ROLE_ADMIN, ROLE_USER):
        raise ValueError(f"Invalid role: {role!r}")
    own = conn is None
    if own:
        conn = _connect()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
            (username, hash_password(password), role),
        )
        if own:
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        if own:
            conn.close()


def verify_user(username, password):
    """Return (True, role) on success, (False, None) otherwise.

    Runs a dummy hash when the user does not exist so response time does
    not leak which usernames are valid.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT password_hash, role FROM users WHERE username = ?",
            ((username or "").strip(),),
        ).fetchone()
    if row is None:
        verify_password(password or "", hash_password("x"))  # constant-time filler
        return False, None
    if verify_password(password or "", row["password_hash"]):
        return True, row["role"]
    return False, None


def get_all_users():
    """Return all users as a list of dicts: id, username, role."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, username, role FROM users ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def count_admins():
    with _connect() as conn:
        cur = conn.execute(
            "SELECT COUNT(*) AS n FROM users WHERE role = ?", (ROLE_ADMIN,))
        return cur.fetchone()["n"]


def delete_user(username):
    """Delete a user by name. Returns True if a row was removed."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM users WHERE username = ?",
                           ((username or "").strip(),))
        conn.commit()
        return cur.rowcount > 0


def set_password(username, new_password):
    """Change a user's password. Returns True on success."""
    if not new_password:
        raise ValueError("Password is required.")
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (hash_password(new_password), (username or "").strip()))
        conn.commit()
        return cur.rowcount > 0
