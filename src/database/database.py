import sqlite3
from pathlib import Path


# ============================================================
# DATABASE LOCATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "safevision.db"
)


# ============================================================
# CREATE DATABASE FOLDER
# ============================================================

DATABASE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# CREATE TABLE
# ============================================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ppe_events (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT NOT NULL,

            camera_id TEXT NOT NULL,

            person_id INTEGER,

            helmet TEXT,

            vest TEXT,

            boots TEXT,

            overall_status TEXT,

            event_type TEXT NOT NULL
        )
        """
    )


    connection.commit()

    connection.close()


# ============================================================
# RUN DATABASE INITIALIZATION
# ============================================================

if __name__ == "__main__":

    initialize_database()

    print("====================================")
    print("SafeVision AI Database")
    print("====================================")
    print("Database created successfully.")
    print()
    print(
        f"Database location:"
    )
    print(DATABASE_PATH)