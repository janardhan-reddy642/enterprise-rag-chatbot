# ============================================================
# AUTHENTICATION + ROLE MANAGEMENT
# SQLite + SHA256
# ADMIN + USER
# CHANGE PASSWORD
# ADMIN RESET PASSWORD
# DELETE USER
# ============================================================

import sqlite3
import hashlib
from pathlib import Path


# ============================================================
# DATABASE PATH
# ============================================================

DB_PATH = Path(__file__).resolve().parent / "users.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


# ============================================================
# PASSWORD HASH
# ============================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db():

    conn = get_connection()
    cursor = conn.cursor()

    # --------------------------------------------------------
    # CREATE USERS TABLE
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
    """)

    # --------------------------------------------------------
    # CHECK EXISTING COLUMNS
    # --------------------------------------------------------

    cursor.execute(
        "PRAGMA table_info(users)"
    )

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    # --------------------------------------------------------
    # ADD ROLE COLUMN FOR OLD DATABASE
    # --------------------------------------------------------

    if "role" not in columns:

        cursor.execute("""
            ALTER TABLE users
            ADD COLUMN role TEXT NOT NULL DEFAULT 'user'
        """)

    # --------------------------------------------------------
    # ADMIN ACCOUNT
    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Admin is created only if it does not exist.
    #
    # Existing admin password will NOT be overwritten.
    #
    # Therefore:
    #
    # Janardhan changes password
    #       ↓
    # Password stays changed
    #       ↓
    # Server restarts
    #       ↓
    # Password still remains changed
    #
    # --------------------------------------------------------

    admin_username = "Janardhan"
    admin_password = "jana@1527"

    cursor.execute("""
        SELECT username
        FROM users
        WHERE username = ?
    """, (
        admin_username,
    ))

    admin_exists = cursor.fetchone()

    # --------------------------------------------------------
    # CREATE ADMIN ONLY IF MISSING
    # --------------------------------------------------------

    if not admin_exists:

        cursor.execute("""
            INSERT INTO users
            (
                username,
                password_hash,
                role
            )
            VALUES (?, ?, ?)
        """, (
            admin_username,
            hash_password(admin_password),
            "admin"
        ))

    else:

        # ----------------------------------------------------
        # IMPORTANT:
        # DO NOT CHANGE ADMIN PASSWORD HERE.
        #
        # Only make sure the role is admin.
        # ----------------------------------------------------

        cursor.execute("""
            UPDATE users
            SET role = 'admin'
            WHERE username = ?
        """, (
            admin_username,
        ))

    conn.commit()
    conn.close()


# ============================================================
# CREATE USER
# ============================================================

def create_user(
    username,
    password,
    role="user"
):

    username = username.strip()

    # --------------------------------------------------------
    # USERNAME VALIDATION
    # --------------------------------------------------------

    if not username:

        return (
            False,
            "Username cannot be empty."
        )

    # --------------------------------------------------------
    # PASSWORD VALIDATION
    # --------------------------------------------------------

    if not password:

        return (
            False,
            "Password cannot be empty."
        )

    # --------------------------------------------------------
    # PASSWORD LENGTH
    # --------------------------------------------------------

    if len(password) < 6:

        return (
            False,
            "Password must contain at least 6 characters."
        )

    # --------------------------------------------------------
    # VALID ROLE
    # --------------------------------------------------------

    if role not in [
        "admin",
        "user"
    ]:

        role = "user"

    conn = get_connection()
    cursor = conn.cursor()

    # --------------------------------------------------------
    # CHECK USERNAME
    # --------------------------------------------------------

    cursor.execute("""
        SELECT username
        FROM users
        WHERE username = ?
    """, (
        username,
    ))

    existing = cursor.fetchone()

    if existing:

        conn.close()

        return (
            False,
            "Username already exists."
        )

    # --------------------------------------------------------
    # CREATE USER
    # --------------------------------------------------------

    cursor.execute("""
        INSERT INTO users
        (
            username,
            password_hash,
            role
        )
        VALUES (?, ?, ?)
    """, (
        username,
        hash_password(password),
        role
    ))

    conn.commit()
    conn.close()

    return (
        True,
        "User created successfully."
    )


# ============================================================
# VERIFY LOGIN
# ============================================================

def verify_user(
    username,
    password
):

    username = username.strip()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT password_hash
        FROM users
        WHERE username = ?
    """, (
        username,
    ))

    row = cursor.fetchone()

    conn.close()

    if not row:

        return False

    stored_hash = row[0]

    return (
        stored_hash
        == hash_password(password)
    )


# ============================================================
# GET USER ROLE
# ============================================================

def get_user_role(username):

    username = username.strip()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT role
        FROM users
        WHERE username = ?
    """, (
        username,
    ))

    row = cursor.fetchone()

    conn.close()

    if not row:

        return None

    return row[0]


# ============================================================
# CHECK ADMIN
# ============================================================

def is_admin(username):

    return (
        get_user_role(username)
        == "admin"
    )


# ============================================================
# CHANGE OWN PASSWORD
# ============================================================

def change_password(
    username,
    old_password,
    new_password
):

    username = username.strip()

    # --------------------------------------------------------
    # NEW PASSWORD VALIDATION
    # --------------------------------------------------------

    if not new_password:

        return (
            False,
            "New password cannot be empty."
        )

    if len(new_password) < 6:

        return (
            False,
            "Password must contain at least 6 characters."
        )

    # --------------------------------------------------------
    # VERIFY OLD PASSWORD
    # --------------------------------------------------------

    if not verify_user(
        username,
        old_password
    ):

        return (
            False,
            "Current password is incorrect."
        )

    # --------------------------------------------------------
    # UPDATE PASSWORD
    # --------------------------------------------------------

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE users
        SET password_hash = ?
        WHERE username = ?
    """, (
        hash_password(new_password),
        username
    ))

    if cursor.rowcount == 0:

        conn.close()

        return (
            False,
            "User not found."
        )

    conn.commit()
    conn.close()

    return (
        True,
        "Password changed successfully."
    )


# ============================================================
# ADMIN RESET PASSWORD
# ============================================================

def admin_reset_password(
    admin_username,
    target_username,
    new_password
):

    admin_username = admin_username.strip()
    target_username = target_username.strip()

    # --------------------------------------------------------
    # CHECK ADMIN
    # --------------------------------------------------------

    if not is_admin(
        admin_username
    ):

        return (
            False,
            "Admin access required."
        )

    # --------------------------------------------------------
    # PASSWORD VALIDATION
    # --------------------------------------------------------

    if not new_password:

        return (
            False,
            "New password cannot be empty."
        )

    if len(new_password) < 6:

        return (
            False,
            "Password must contain at least 6 characters."
        )

    # --------------------------------------------------------
    # CHECK TARGET USER
    # --------------------------------------------------------

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT username
        FROM users
        WHERE username = ?
    """, (
        target_username,
    ))

    user = cursor.fetchone()

    if not user:

        conn.close()

        return (
            False,
            "User not found."
        )

    # --------------------------------------------------------
    # RESET PASSWORD
    # --------------------------------------------------------

    cursor.execute("""
        UPDATE users
        SET password_hash = ?
        WHERE username = ?
    """, (
        hash_password(new_password),
        target_username
    ))

    conn.commit()
    conn.close()

    return (
        True,
        "Password reset successfully."
    )


# ============================================================
# GET ALL USERS
# ============================================================

def get_all_users():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT username, role
        FROM users
        ORDER BY username
    """)

    users = cursor.fetchall()

    conn.close()

    return users


# ============================================================
# DELETE USER
# ============================================================

def delete_user(
    admin_username,
    target_username
):

    admin_username = admin_username.strip()
    target_username = target_username.strip()

    # --------------------------------------------------------
    # CHECK ADMIN
    # --------------------------------------------------------

    if not is_admin(
        admin_username
    ):

        return (
            False,
            "Admin access required."
        )

    # --------------------------------------------------------
    # ADMIN CANNOT DELETE SELF
    # --------------------------------------------------------

    if (
        admin_username
        == target_username
    ):

        return (
            False,
            "Admin cannot delete their own account."
        )

    # --------------------------------------------------------
    # DELETE TARGET USER
    # --------------------------------------------------------

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM users
        WHERE username = ?
    """, (
        target_username,
    ))

    if cursor.rowcount == 0:

        conn.close()

        return (
            False,
            "User not found."
        )

    conn.commit()
    conn.close()

    return (
        True,
        "User deleted successfully."
    )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

init_db()