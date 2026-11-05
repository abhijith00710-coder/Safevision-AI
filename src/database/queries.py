from .database import get_connection


# ============================================================
# BASIC COUNTS
# ============================================================

def get_total_event_count():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM ppe_events
    """)

    result = cursor.fetchone()[0]

    connection.close()

    return result


def get_unique_worker_count():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(DISTINCT person_id)
        FROM ppe_events
        WHERE person_id IS NOT NULL
    """)

    result = cursor.fetchone()[0]

    connection.close()

    return result


# ============================================================
# RECENT EVENTS
# ============================================================

def get_recent_ppe_events(limit=20):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM ppe_events
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    events = cursor.fetchall()

    connection.close()

    return events


# ============================================================
# PERSON HISTORY
# ============================================================

def get_person_events(person_id):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            timestamp,
            camera_id,
            person_id,
            helmet,
            vest,
            boots,
            overall_status,
            event_type
        FROM ppe_events
        WHERE person_id = ?
        ORDER BY id DESC
    """, (person_id,))

    events = cursor.fetchall()

    connection.close()

    return events


# ============================================================
# LATEST STATUS OF EVERY WORKER
# ============================================================

def get_latest_worker_status():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            p.id,
            p.timestamp,
            p.camera_id,
            p.person_id,
            p.helmet,
            p.vest,
            p.boots,
            p.overall_status,
            p.event_type
        FROM ppe_events p

        INNER JOIN (
            SELECT
                person_id,
                MAX(id) AS latest_id
            FROM ppe_events
            WHERE person_id IS NOT NULL
            GROUP BY person_id
        ) latest

        ON p.id = latest.latest_id

        ORDER BY p.person_id
    """)

    events = cursor.fetchall()

    connection.close()

    return events


# ============================================================
# LATEST COMPLIANT WORKERS
# ============================================================

def get_latest_compliant_workers():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            p.person_id,
            p.timestamp,
            p.helmet,
            p.vest,
            p.boots,
            p.overall_status
        FROM ppe_events p

        INNER JOIN (
            SELECT
                person_id,
                MAX(id) AS latest_id
            FROM ppe_events
            WHERE person_id IS NOT NULL
            GROUP BY person_id
        ) latest

        ON p.id = latest.latest_id

        WHERE p.overall_status = 'COMPLIANT'

        ORDER BY p.person_id
    """)

    results = cursor.fetchall()

    connection.close()

    return results


# ============================================================
# LATEST NON-COMPLIANT WORKERS
# ============================================================

def get_latest_non_compliant_workers():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            p.person_id,
            p.timestamp,
            p.helmet,
            p.vest,
            p.boots,
            p.overall_status
        FROM ppe_events p

        INNER JOIN (
            SELECT
                person_id,
                MAX(id) AS latest_id
            FROM ppe_events
            WHERE person_id IS NOT NULL
            GROUP BY person_id
        ) latest

        ON p.id = latest.latest_id

        WHERE p.overall_status = 'NON-COMPLIANT'

        ORDER BY p.person_id
    """)

    results = cursor.fetchall()

    connection.close()

    return results


# ============================================================
# LATEST UNKNOWN WORKERS
# ============================================================

def get_latest_unknown_workers():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            p.person_id,
            p.timestamp,
            p.helmet,
            p.vest,
            p.boots,
            p.overall_status
        FROM ppe_events p

        INNER JOIN (
            SELECT
                person_id,
                MAX(id) AS latest_id
            FROM ppe_events
            WHERE person_id IS NOT NULL
            GROUP BY person_id
        ) latest

        ON p.id = latest.latest_id

        WHERE p.overall_status = 'UNKNOWN'

        ORDER BY p.person_id
    """)

    results = cursor.fetchall()

    connection.close()

    return results


# ============================================================
# HISTORICAL NON-COMPLIANT EVENTS
# ============================================================

def get_non_compliant_events():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM ppe_events
        WHERE overall_status = 'NON-COMPLIANT'
        ORDER BY id DESC
    """)

    events = cursor.fetchall()

    connection.close()

    return events


# ============================================================
# HISTORICAL COMPLIANT EVENTS
# ============================================================

def get_compliant_events():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM ppe_events
        WHERE overall_status = 'COMPLIANT'
        ORDER BY id DESC
    """)

    events = cursor.fetchall()

    connection.close()

    return events


# ============================================================
# HISTORICAL UNKNOWN EVENTS
# ============================================================

def get_unknown_events():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM ppe_events
        WHERE overall_status = 'UNKNOWN'
        ORDER BY id DESC
    """)

    events = cursor.fetchall()

    connection.close()

    return events


# ============================================================
# LATEST WORKERS WITHOUT HELMET
# ============================================================

def get_workers_without_helmet():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            p.person_id,
            p.timestamp,
            p.helmet,
            p.vest,
            p.boots,
            p.overall_status
        FROM ppe_events p

        INNER JOIN (
            SELECT
                person_id,
                MAX(id) AS latest_id
            FROM ppe_events
            WHERE person_id IS NOT NULL
            GROUP BY person_id
        ) latest

        ON p.id = latest.latest_id

        WHERE p.helmet = 'NO'

        ORDER BY p.person_id
    """)

    results = cursor.fetchall()

    connection.close()

    return results


# ============================================================
# COUNT CURRENT WORKERS WITHOUT HELMET
# ============================================================

def get_count_without_helmet():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM ppe_events p

        INNER JOIN (
            SELECT
                person_id,
                MAX(id) AS latest_id
            FROM ppe_events
            WHERE person_id IS NOT NULL
            GROUP BY person_id
        ) latest

        ON p.id = latest.latest_id

        WHERE p.helmet = 'NO'
    """)

    result = cursor.fetchone()[0]

    connection.close()

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("====================================")
    print("SafeVision AI Database Queries Test")
    print("====================================")

    print()

    print(
        "Total PPE events:",
        get_total_event_count()
    )

    print(
        "Unique workers:",
        get_unique_worker_count()
    )

    print(
        "Current workers without helmet:",
        get_count_without_helmet()
    )

    print()

    print("Latest worker statuses:")
    print("------------------------------------")

    workers = get_latest_worker_status()

    for worker in workers:

        print(
            f"Person ID: {worker['person_id']} | "
            f"Helmet: {worker['helmet']} | "
            f"Vest: {worker['vest']} | "
            f"Boots: {worker['boots']} | "
            f"Overall: {worker['overall_status']} | "
            f"Timestamp: {worker['timestamp']}"
        )

    print()

    print("====================================")