import sqlite3
from datetime import datetime

try:
    from .database import get_connection
except ImportError:
    from database import get_connection


# ============================================================
# INSERT PPE EVENT
# ============================================================

def log_ppe_event(
    camera_id,
    person_id,
    helmet,
    vest,
    boots,
    overall_status,
    event_type
):

    connection = get_connection()

    cursor = connection.cursor()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute(
        """
        INSERT INTO ppe_events (
            timestamp,
            camera_id,
            person_id,
            helmet,
            vest,
            boots,
            overall_status,
            event_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp,
            camera_id,
            person_id,
            helmet,
            vest,
            boots,
            overall_status,
            event_type
        )
    )

    connection.commit()

    connection.close()


# ============================================================
# RETRIEVE EVENTS
# ============================================================

def get_recent_events(limit=10):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM ppe_events
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    events = cursor.fetchall()

    connection.close()

    return events


# ============================================================
# TEST DATABASE
# ============================================================

if __name__ == "__main__":

    print("====================================")
    print("SafeVision AI Logger Test")
    print("====================================")

    # Insert one test event

    log_ppe_event(
        camera_id="CAM_01",
        person_id=1,
        helmet="YES",
        vest="YES",
        boots="UNKNOWN",
        overall_status="UNKNOWN",
        event_type="PPE_CHECK"
    )

    print("Test event inserted successfully.")
    print()

    # Retrieve event

    events = get_recent_events()

    print("Recent events:")
    print("------------------------------------")

    for event in events:

        print(
            f"ID: {event['id']} | "
            f"Time: {event['timestamp']} | "
            f"Camera: {event['camera_id']} | "
            f"Person: {event['person_id']} | "
            f"Helmet: {event['helmet']} | "
            f"Vest: {event['vest']} | "
            f"Boots: {event['boots']} | "
            f"Overall: {event['overall_status']} | "
            f"Event: {event['event_type']}"
        )