import re

from src.database.queries import (
    get_total_event_count,
    get_unique_worker_count,
    get_recent_ppe_events,
    get_person_events,
    get_latest_worker_status,
    get_latest_compliant_workers,
    get_latest_non_compliant_workers,
    get_latest_unknown_workers,
    get_workers_without_helmet,
    get_count_without_helmet,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def row_value(row, key, index=None):
    """
    Safely read a value from sqlite3.Row or tuple.
    """

    try:
        return row[key]
    except (KeyError, TypeError, IndexError):

        if index is not None:
            try:
                return row[index]
            except (IndexError, TypeError):
                pass

    return None


# ============================================================
# FORMAT ONE WORKER
# ============================================================

def format_worker_status(worker):
    """
    Format a worker record.

    IMPORTANT:
    person_id is the worker identifier.
    id is the database event ID and must NOT be called worker ID.
    """

    person_id = row_value(
        worker,
        "person_id",
        3
    )

    timestamp = row_value(
        worker,
        "timestamp",
        1
    )

    camera_id = row_value(
        worker,
        "camera_id",
        2
    )

    helmet = row_value(
        worker,
        "helmet",
        4
    )

    vest = row_value(
        worker,
        "vest",
        5
    )

    boots = row_value(
        worker,
        "boots",
        6
    )

    overall = row_value(
        worker,
        "overall_status",
        7
    )

    event_type = row_value(
        worker,
        "event_type",
        8
    )

    return (
        f"Person ID: {person_id} | "
        f"Helmet: {helmet} | "
        f"Vest: {vest} | "
        f"Boots: {boots} | "
        f"Overall: {overall} | "
        f"Timestamp: {timestamp} | "
        f"Camera: {camera_id} | "
        f"Event: {event_type}"
    )


# ============================================================
# FORMAT CURRENT WORKER LIST
# ============================================================

def format_worker_list(workers):

    if not workers:
        return "No matching workers were found."

    lines = []

    for worker in workers:

        person_id = row_value(
            worker,
            "person_id",
            0
        )

        timestamp = row_value(
            worker,
            "timestamp",
            1
        )

        helmet = row_value(
            worker,
            "helmet",
            2
        )

        vest = row_value(
            worker,
            "vest",
            3
        )

        boots = row_value(
            worker,
            "boots",
            4
        )

        overall = row_value(
            worker,
            "overall_status",
            5
        )

        lines.append(
            f"Person ID: {person_id} | "
            f"Helmet: {helmet} | "
            f"Vest: {vest} | "
            f"Boots: {boots} | "
            f"Overall: {overall} | "
            f"Timestamp: {timestamp}"
        )

    return "\n".join(lines)


# ============================================================
# FORMAT RECENT EVENTS
# ============================================================

def format_recent_events(events):

    if not events:
        return "No PPE events were found."

    lines = []

    for event in events:

        event_id = row_value(
            event,
            "id",
            0
        )

        timestamp = row_value(
            event,
            "timestamp",
            1
        )

        camera_id = row_value(
            event,
            "camera_id",
            2
        )

        person_id = row_value(
            event,
            "person_id",
            3
        )

        helmet = row_value(
            event,
            "helmet",
            4
        )

        vest = row_value(
            event,
            "vest",
            5
        )

        boots = row_value(
            event,
            "boots",
            6
        )

        overall = row_value(
            event,
            "overall_status",
            7
        )

        event_type = row_value(
            event,
            "event_type",
            8
        )

        lines.append(
            f"Event ID: {event_id} | "
            f"Person ID: {person_id} | "
            f"Helmet: {helmet} | "
            f"Vest: {vest} | "
            f"Boots: {boots} | "
            f"Overall: {overall} | "
            f"Timestamp: {timestamp} | "
            f"Event: {event_type}"
        )

    return "\n".join(lines)


# ============================================================
# SPECIFIC PERSON
# ============================================================

def get_worker_status(person_id):

    events = get_person_events(person_id)

    if not events:
        return (
            f"No database record was found for "
            f"Person ID {person_id}."
        )

    # get_person_events() is already ordered by newest event first.
    latest = events[0]

    event_id = row_value(
        latest,
        "id",
        0
    )

    timestamp = row_value(
        latest,
        "timestamp",
        1
    )

    camera_id = row_value(
        latest,
        "camera_id",
        2
    )

    stored_person_id = row_value(
        latest,
        "person_id",
        3
    )

    helmet = row_value(
        latest,
        "helmet",
        4
    )

    vest = row_value(
        latest,
        "vest",
        5
    )

    boots = row_value(
        latest,
        "boots",
        6
    )

    overall = row_value(
        latest,
        "overall_status",
        7
    )

    event_type = row_value(
        latest,
        "event_type",
        8
    )

    return (
        f"Person ID: {stored_person_id}\n"
        f"Helmet: {helmet}\n"
        f"Vest: {vest}\n"
        f"Boots: {boots}\n"
        f"Overall Status: {overall}\n"
        f"Timestamp: {timestamp}\n"
        f"Camera: {camera_id}\n"
        f"Event ID: {event_id}\n"
        f"Event Type: {event_type}"
    )


# ============================================================
# MAIN DATABASE QUESTION ROUTER
# ============================================================

def answer_database_question(question):

    question_lower = question.lower().strip()

    # --------------------------------------------------------
    # SPECIFIC PERSON ID
    # --------------------------------------------------------

    person_match = re.search(
        r"(?:person|worker)\s*(?:id)?\s*#?\s*(\d+)",
        question_lower
    )

    # --------------------------------------------------------
    # TOTAL EVENTS
    # --------------------------------------------------------

    if (
        "how many events" in question_lower
        or "total events" in question_lower
        or "number of events" in question_lower
    ):

        count = get_total_event_count()

        return (
            f"The database contains "
            f"{count} PPE events."
        )

    # --------------------------------------------------------
    # TOTAL / UNIQUE WORKERS
    # --------------------------------------------------------

    if (
        "how many workers" in question_lower
        or "number of workers" in question_lower
        or "workers detected" in question_lower
        or "how many people" in question_lower
    ):

        # Do not accidentally treat entry/exit questions as
        # simple worker-count questions.
        if (
            "entered" not in question_lower
            and "entry" not in question_lower
            and "exit" not in question_lower
            and "exited" not in question_lower
        ):

            count = get_unique_worker_count()

            return (
                f"The database contains "
                f"{count} unique worker/person IDs."
            )

    # --------------------------------------------------------
    # CURRENT COMPLIANT WORKERS
    # --------------------------------------------------------

    if (
        "which workers are compliant" in question_lower
        or "which workers are currently compliant" in question_lower
        or "who is compliant" in question_lower
        or "who are compliant" in question_lower
    ):

        workers = get_latest_compliant_workers()

        if not workers:
            return (
                "No workers currently have a confirmed "
                "COMPLIANT status in the available "
                "database records."
            )

        lines = [
            "Currently compliant workers:"
        ]

        for worker in workers:

            person_id = row_value(
                worker,
                "person_id",
                0
            )

            timestamp = row_value(
                worker,
                "timestamp",
                1
            )

            lines.append(
                f"Person ID: {person_id} | "
                f"Status: COMPLIANT | "
                f"Timestamp: {timestamp}"
            )

        return "\n".join(lines)

    # --------------------------------------------------------
    # CURRENT NON-COMPLIANT WORKERS
    # --------------------------------------------------------

    if (
        "which workers are non-compliant" in question_lower
        or "which workers are currently non-compliant"
        in question_lower
        or "who is non-compliant" in question_lower
        or "who are non-compliant" in question_lower
    ):

        workers = get_latest_non_compliant_workers()

        if not workers:
            return (
                "No workers currently have a confirmed "
                "NON-COMPLIANT status in the available "
                "database records."
            )

        lines = [
            "Currently non-compliant workers:"
        ]

        for worker in workers:

            person_id = row_value(
                worker,
                "person_id",
                0
            )

            timestamp = row_value(
                worker,
                "timestamp",
                1
            )

            helmet = row_value(
                worker,
                "helmet",
                2
            )

            vest = row_value(
                worker,
                "vest",
                3
            )

            boots = row_value(
                worker,
                "boots",
                4
            )

            lines.append(
                f"Person ID: {person_id} | "
                f"Helmet: {helmet} | "
                f"Vest: {vest} | "
                f"Boots: {boots} | "
                f"Status: NON-COMPLIANT | "
                f"Timestamp: {timestamp}"
            )

        return "\n".join(lines)

    # --------------------------------------------------------
    # CURRENT UNKNOWN WORKERS
    # --------------------------------------------------------

    if (
        "which workers are unknown" in question_lower
        or "which workers have unknown" in question_lower
        or "who is unknown" in question_lower
        or "unknown workers" in question_lower
    ):

        workers = get_latest_unknown_workers()

        if not workers:
            return (
                "No workers currently have an "
                "UNKNOWN overall status."
            )

        lines = [
            "Currently unknown workers:"
        ]

        for worker in workers:

            person_id = row_value(
                worker,
                "person_id",
                0
            )

            timestamp = row_value(
                worker,
                "timestamp",
                1
            )

            helmet = row_value(
                worker,
                "helmet",
                2
            )

            vest = row_value(
                worker,
                "vest",
                3
            )

            boots = row_value(
                worker,
                "boots",
                4
            )

            lines.append(
                f"Person ID: {person_id} | "
                f"Helmet: {helmet} | "
                f"Vest: {vest} | "
                f"Boots: {boots} | "
                f"Status: UNKNOWN | "
                f"Timestamp: {timestamp}"
            )

        return "\n".join(lines)

    # --------------------------------------------------------
    # COUNT CURRENT COMPLIANT
    # --------------------------------------------------------

    if (
        "how many workers are compliant" in question_lower
        or "how many are compliant" in question_lower
        or "number of compliant workers" in question_lower
    ):

        workers = get_latest_compliant_workers()

        return (
            f"There are currently "
            f"{len(workers)} compliant workers "
            f"in the latest available worker records."
        )

    # --------------------------------------------------------
    # COUNT CURRENT NON-COMPLIANT
    # --------------------------------------------------------

    if (
        "how many workers are non-compliant"
        in question_lower
        or "how many are non-compliant"
        in question_lower
        or "number of non-compliant workers"
        in question_lower
    ):

        workers = get_latest_non_compliant_workers()

        return (
            f"There are currently "
            f"{len(workers)} non-compliant workers "
            f"in the latest available worker records."
        )

    # --------------------------------------------------------
    # COUNT CURRENT UNKNOWN
    # --------------------------------------------------------

    if (
        "how many workers are unknown" in question_lower
        or "how many are unknown" in question_lower
        or "number of unknown workers" in question_lower
    ):

        workers = get_latest_unknown_workers()

        return (
            f"There are currently "
            f"{len(workers)} workers with UNKNOWN "
            f"overall status in the latest available "
            f"worker records."
        )

    # --------------------------------------------------------
    # WORKERS WITHOUT HELMET
    # --------------------------------------------------------

    if (
        "without helmet" in question_lower
        or "without a helmet" in question_lower
        or "no helmet" in question_lower
        or "don't have helmet" in question_lower
        or "do not have helmet" in question_lower
        or "doesn't have helmet" in question_lower
    ):

        # Entry/exit is deliberately not supported.
        if (
            "entered" in question_lower
            or "entry" in question_lower
            or "exited" in question_lower
            or "exit" in question_lower
        ):

            return (
                "Entry/Exit tracking is not implemented "
                "in the current SafeVision AI application. "
                "The database can report current PPE status "
                "but cannot determine how many workers "
                "entered or exited."
            )

        workers = get_workers_without_helmet()

        if not workers:
            return (
                "No current worker records have a confirmed "
                "NO helmet status."
            )

        lines = [
            "Workers with a currently confirmed missing helmet:"
        ]

        for worker in workers:

            person_id = row_value(
                worker,
                "person_id",
                0
            )

            timestamp = row_value(
                worker,
                "timestamp",
                1
            )

            helmet = row_value(
                worker,
                "helmet",
                2
            )

            vest = row_value(
                worker,
                "vest",
                3
            )

            boots = row_value(
                worker,
                "boots",
                4
            )

            overall = row_value(
                worker,
                "overall_status",
                5
            )

            lines.append(
                f"Person ID: {person_id} | "
                f"Helmet: {helmet} | "
                f"Vest: {vest} | "
                f"Boots: {boots} | "
                f"Overall: {overall} | "
                f"Timestamp: {timestamp}"
            )

        return "\n".join(lines)

    # --------------------------------------------------------
    # SPECIFIC PERSON
    # --------------------------------------------------------

    if person_match:

        person_id = int(
            person_match.group(1)
        )

        return get_worker_status(
            person_id
        )

    # --------------------------------------------------------
    # RECENT EVENTS
    # --------------------------------------------------------

    if (
        "recent events" in question_lower
        or "latest events" in question_lower
        or "recent ppe events" in question_lower
        or "show recent events" in question_lower
    ):

        events = get_recent_ppe_events(
            limit=10
        )

        return format_recent_events(
            events
        )

    # --------------------------------------------------------
    # CURRENT WORKER STATUS
    # --------------------------------------------------------

    if (
        "current worker status" in question_lower
        or "current workers" in question_lower
        or "latest worker status" in question_lower
    ):

        workers = get_latest_worker_status()

        if not workers:
            return (
                "No current worker status records "
                "were found."
            )

        return format_worker_list(
            workers
        )

    # --------------------------------------------------------
    # ENTRY / EXIT
    # --------------------------------------------------------

    if (
        "entered" in question_lower
        or "entry" in question_lower
        or "exited" in question_lower
        or "exit" in question_lower
    ):

        return (
            "Entry/Exit tracking is not implemented "
            "in the current SafeVision AI application."
        )

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return (
        "The database assistant could not identify "
        "a supported database query. You can ask about "
        "current worker status, compliant workers, "
        "non-compliant workers, unknown workers, "
        "helmet status, a specific Person ID, or "
        "recent PPE events."
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print("SafeVision AI Database Assistant")
    print("==========================================")

    test_questions = [
        "Which workers are currently compliant?",
        "Which workers are currently non-compliant?",
        "How many workers are compliant?",
        "How many workers are non-compliant?",
        "What is the PPE status of Person 148?",
    ]

    for question in test_questions:

        print()
        print("------------------------------------------")
        print("QUESTION:")
        print(question)
        print("------------------------------------------")

        answer = answer_database_question(
            question
        )

        print(answer)

    print()
    print("==========================================")
    print("Test completed.")
    print("==========================================")