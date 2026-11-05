import streamlit as st
import cv2
import time

from pathlib import Path
from ultralytics import YOLO

from src.database.logger import log_ppe_event


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "train"
    / "weights"
    / "best.pt"
)

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / "CAM_01"
    / "test_.mp4"
)


# ============================================================
# SETTINGS
# ============================================================

DETECTION_CONFIDENCE = 0.40
ASSOCIATION_THRESHOLD = 0.40

# Number of consecutive YES detections required
# to confirm PPE presence.
CONFIRM_FRAMES = 3

# Keep a confirmed PPE status stable when the detector
# temporarily loses the PPE item.
PPE_STABILITY_SECONDS = 60

# A confirmed NO condition must remain continuously
# observed for this long before changing to NO.
NON_COMPLIANT_CONFIRM_SECONDS = 10

# Remove a worker from memory after not seeing them
# for this many frames.
PERSON_MEMORY_FRAMES = 180

CAMERA_ID = "CAM_01"


# ============================================================
# PPE CLASSES
# ============================================================

# PPE currently monitored by SafeVision AI.
POSITIVE_CLASSES = {
    "helmet": "helmet",
    "vest": "vest",
    "boots": "boots",
}

# Negative classes currently available in the model.
NEGATIVE_CLASSES = {
    "no_helmet": "helmet",
    "no_boots": "boots",
}

PPE_TYPES = [
    "helmet",
    "vest",
    "boots"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return YOLO(str(MODEL_PATH))


# ============================================================
# BODY REGION
# ============================================================

def get_body_region(person_box, ppe_box):

    px1, py1, px2, py2 = person_box
    x1, y1, x2, y2 = ppe_box

    person_width = px2 - px1
    person_height = py2 - py1

    if person_width <= 0 or person_height <= 0:
        return None, 0.0

    ppe_center_x = (x1 + x2) / 2
    ppe_center_y = (y1 + y2) / 2

    nx = (ppe_center_x - px1) / person_width
    ny = (ppe_center_y - py1) / person_height

    expanded_x1 = px1 - person_width * 0.25
    expanded_x2 = px2 + person_width * 0.25

    expanded_y1 = py1 - person_height * 0.15
    expanded_y2 = py2 + person_height * 0.15

    if not (
        expanded_x1 <= ppe_center_x <= expanded_x2
        and
        expanded_y1 <= ppe_center_y <= expanded_y2
    ):
        return None, 0.0

    # --------------------------------------------------------
    # HEAD
    # --------------------------------------------------------

    if 0.00 <= ny <= 0.30:

        distance = abs(nx - 0.50)

        score = 1.0 - min(
            distance / 0.50,
            1.0
        )

        return "head", score

    # --------------------------------------------------------
    # TORSO
    # --------------------------------------------------------

    if 0.20 <= ny <= 0.70:

        horizontal_score = 1.0 - min(
            abs(nx - 0.50) / 0.60,
            1.0
        )

        vertical_score = 1.0 - min(
            abs(ny - 0.45) / 0.50,
            1.0
        )

        score = (
            horizontal_score * 0.6
            +
            vertical_score * 0.4
        )

        return "torso", score

    # --------------------------------------------------------
    # FEET
    # --------------------------------------------------------

    if 0.65 <= ny <= 1.15:

        vertical_score = 1.0 - min(
            abs(ny - 0.90) / 0.40,
            1.0
        )

        return "feet", max(0.0, vertical_score)

    return None, 0.0


# ============================================================
# PPE → PERSON ASSOCIATION
# ============================================================

def calculate_association_score(
    person_box,
    ppe_box,
    ppe_type
):

    region, score = get_body_region(
        person_box,
        ppe_box
    )

    expected_regions = {
        "helmet": ["head"],
        "vest": ["torso"],
        "boots": ["feet"]
    }

    if region not in expected_regions.get(
        ppe_type,
        []
    ):
        return 0.0

    return score


# ============================================================
# CREATE PERSON MEMORY
# ============================================================

def create_person_memory(frame_number):

    return {
        "last_seen": frame_number,

        # ----------------------------------------------------
        # PPE STATES
        # ----------------------------------------------------

        "helmet": {
            "status": "UNKNOWN",

            # YES confirmation
            "yes_count": 0,

            # NO confirmation timer
            "no_since": None,

            # Last time YES was actually observed
            "last_yes_time": None,
        },

        "vest": {
            "status": "UNKNOWN",
            "yes_count": 0,
            "no_since": None,
            "last_yes_time": None,
        },

        "boots": {
            "status": "UNKNOWN",
            "yes_count": 0,
            "no_since": None,
            "last_yes_time": None,
        },

        # Used to avoid unnecessary database logging.
        "last_logged_status": None,
    }


# ============================================================
# UPDATE PPE MEMORY
# ============================================================

def update_ppe_memory(
    memory,
    observation,
    current_time
):
    """
    Updates the status of one PPE item.

    YES:
        Requires CONFIRM_FRAMES consecutive observations.

    NO:
        Requires NON_COMPLIANT_CONFIRM_SECONDS continuous
        confirmation.

    None:
        Temporary missed detection.
        Existing confirmed YES remains stable for
        PPE_STABILITY_SECONDS.
    """

    # ========================================================
    # YES OBSERVATION
    # ========================================================

    if observation == "YES":

        memory["yes_count"] += 1

        # A fresh YES cancels any pending NO timer.
        memory["no_since"] = None

        # Confirm YES after required number of frames.
        if memory["yes_count"] >= CONFIRM_FRAMES:

            memory["status"] = "YES"

            memory["last_yes_time"] = current_time

        return memory["status"]

    # ========================================================
    # NO OBSERVATION
    # ========================================================

    if observation == "NO":

        # Reset YES confirmation.
        memory["yes_count"] = 0

        # Start NO timer.
        if memory["no_since"] is None:

            memory["no_since"] = current_time

        no_duration = (
            current_time
            -
            memory["no_since"]
        )

        # Confirm NO only after 10 seconds.
        if (
            no_duration
            >= NON_COMPLIANT_CONFIRM_SECONDS
        ):

            memory["status"] = "NO"

        return memory["status"]

    # ========================================================
    # NO CURRENT OBSERVATION
    # ========================================================

    # Do not immediately change the confirmed status.
    #
    # This prevents:
    #
    # detector temporarily misses helmet
    #          ↓
    # helmet becomes UNKNOWN
    #
    # Instead, retain YES for the stability period.

    if (
        memory["status"] == "YES"
        and
        memory["last_yes_time"] is not None
    ):

        elapsed = (
            current_time
            -
            memory["last_yes_time"]
        )

        if elapsed <= PPE_STABILITY_SECONDS:

            return "YES"

        # The PPE has not been positively observed
        # for more than the stability period.
        memory["status"] = "UNKNOWN"

    return memory["status"]


# ============================================================
# OVERALL PPE STATUS
# ============================================================

def calculate_overall_status(ppe):

    values = [
        ppe["helmet"],
        ppe["vest"],
        ppe["boots"]
    ]

    # Any confirmed missing PPE means
    # NON-COMPLIANT.
    if "NO" in values:

        return "NON-COMPLIANT"

    # All monitored PPE confirmed present.
    if all(
        value == "YES"
        for value in values
    ):

        return "COMPLIANT"

    # Otherwise evidence is incomplete.
    return "UNKNOWN"


# ============================================================
# LOG STATUS ONLY WHEN IT REALLY CHANGES
# ============================================================

def log_status_if_changed(
    person_id,
    ppe,
    overall,
    person_memory
):

    current_status = (
        ppe["helmet"],
        ppe["vest"],
        ppe["boots"],
        overall
    )

    previous_status = (
        person_memory.get(
            "last_logged_status"
        )
    )

    # --------------------------------------------------------
    # Do not repeatedly log identical states.
    # --------------------------------------------------------

    if previous_status == current_status:

        return

    # --------------------------------------------------------
    # Avoid creating database records for temporary UNKNOWN
    # states.
    #
    # UNKNOWN is still displayed in the UI, but we don't want
    # the database flooded by temporary detection changes.
    # --------------------------------------------------------

    if overall == "UNKNOWN":

        # If there was no previous meaningful state,
        # don't log UNKNOWN.
        if previous_status is None:

            return

        # If we already have a meaningful state,
        # don't overwrite it with temporary UNKNOWN.
        return

    # --------------------------------------------------------
    # Save meaningful confirmed status.
    # --------------------------------------------------------

    try:

        log_ppe_event(
            camera_id=CAMERA_ID,
            person_id=int(person_id),
            helmet=ppe["helmet"],
            vest=ppe["vest"],
            boots=ppe["boots"],
            overall_status=overall,
            event_type="PPE_CHECK"
        )

        person_memory[
            "last_logged_status"
        ] = current_status

    except Exception as error:

        print(
            f"Database error: {error}"
        )


# ============================================================
# STREAMLIT MONITOR
# ============================================================

@st.fragment(run_every="5s")
def show_live_monitor():

    st.subheader(
        "🎥 Live PPE Monitoring"
    )

    st.caption(
        "Stable PPE monitoring with ByteTrack, "
        "60-second detection stability and "
        "10-second non-compliance confirmation."
    )

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not MODEL_PATH.exists():

        st.error(
            f"YOLO model not found:\n\n{MODEL_PATH}"
        )

        return

    # ========================================================
    # CHECK VIDEO
    # ========================================================

    if not VIDEO_PATH.exists():

        st.error(
            f"Video not found:\n\n{VIDEO_PATH}"
        )

        return

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()

    st.success(
        "YOLO PPE model loaded successfully."
    )

    # ========================================================
    # PERSISTENT MONITORING STATE
    # ========================================================

    if "sv_monitoring" not in st.session_state:
        st.session_state.sv_monitoring = False

    if "sv_cap" not in st.session_state:
        st.session_state.sv_cap = None

    if "sv_person_memory" not in st.session_state:
        st.session_state.sv_person_memory = {}

    if "sv_frame_number" not in st.session_state:
        st.session_state.sv_frame_number = 0

    # ========================================================
    # MONITOR CONTROLS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "▶️ Start PPE Detection",
            type="primary",
            width="stretch"
        ):

            if st.session_state.sv_cap is None:

                cap = cv2.VideoCapture(
                    str(VIDEO_PATH)
                )

                if not cap.isOpened():

                    st.error(
                        "Unable to open video."
                    )

                    return

                st.session_state.sv_cap = cap

            st.session_state.sv_monitoring = True

    with col2:

        if st.button(
            "⏹️ Stop Monitoring",
            width="stretch"
        ):

            st.session_state.sv_monitoring = False

            if st.session_state.sv_cap is not None:

                st.session_state.sv_cap.release()

            st.session_state.sv_cap = None

            st.session_state.sv_person_memory = {}

            st.session_state.sv_frame_number = 0

    # ========================================================
    # WAITING STATE
    # ========================================================

    if not st.session_state.sv_monitoring:

        st.info(
            "Click **Start PPE Detection** to begin monitoring."
        )

        return

    # ========================================================
    # ACTIVE MONITORING STATE
    # ========================================================

    cap = st.session_state.sv_cap
    person_memory = st.session_state.sv_person_memory

    if cap is None:

        st.session_state.sv_monitoring = False

        st.warning(
            "Monitoring video is not available."
        )

        return

    # ========================================================
    # STREAMLIT UI
    # ========================================================

    video_placeholder = st.empty()

    status_placeholder = st.empty()

    st.divider()

    st.subheader(
        "👷 Current Worker Status"
    )

    worker_placeholder = st.empty()

    # ========================================================
    # PROCESS A SMALL BATCH OF FRAMES
    # ========================================================

    for _ in range(45):

        ret, frame = cap.read()

        if not ret:

            st.session_state.sv_monitoring = False

            cap.release()

            st.session_state.sv_cap = None

            st.success(
                "PPE monitoring completed."
            )

            return

        st.session_state.sv_frame_number += 1

        frame_number = st.session_state.sv_frame_number

        current_time = time.monotonic()

        # ====================================================
        # YOLO + BYTE TRACK
        # ====================================================

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=DETECTION_CONFIDENCE,
            verbose=False
        )

        result = results[0]

        persons = []

        ppe_detections = []

        # ====================================================
        # READ DETECTIONS
        # ====================================================

        if result.boxes is not None:

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            class_ids = (
                result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            if result.boxes.id is not None:

                track_ids = (
                    result.boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

            else:

                track_ids = [
                    None
                ] * len(boxes)

            # =================================================
            # SEPARATE PERSON / PPE
            # =================================================

            for (
                box,
                class_id,
                confidence,
                track_id
            ) in zip(
                boxes,
                class_ids,
                confidences,
                track_ids
            ):

                x1, y1, x2, y2 = (
                    box.astype(int)
                )

                class_name = model.names[
                    class_id
                ].lower()

                # ---------------------------------------------
                # PERSON
                # ---------------------------------------------

                if class_name == "person":

                    persons.append({

                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                        "id": track_id,

                        "current_ppe": {
                            "helmet": None,
                            "vest": None,
                            "boots": None
                        }
                    })

                # ---------------------------------------------
                # POSITIVE PPE
                # ---------------------------------------------

                elif class_name in POSITIVE_CLASSES:

                    ppe_detections.append({

                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                        "type": POSITIVE_CLASSES[
                            class_name
                        ],

                        "status": "YES",

                        "confidence": confidence,

                        "class_name": class_name
                    })

                # ---------------------------------------------
                # NEGATIVE PPE
                # ---------------------------------------------

                elif class_name in NEGATIVE_CLASSES:

                    ppe_detections.append({

                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                        "type": NEGATIVE_CLASSES[
                            class_name
                        ],

                        "status": "NO",

                        "confidence": confidence,

                        "class_name": class_name
                    })

        # ====================================================
        # UPDATE PERSON MEMORY
        # ====================================================

        for person in persons:

            person_id = person["id"]

            if person_id is None:

                continue

            if person_id not in person_memory:

                person_memory[
                    person_id
                ] = create_person_memory(
                    frame_number
                )

            person_memory[
                person_id
            ][
                "last_seen"
            ] = frame_number

        # ====================================================
        # ASSOCIATE PPE WITH PERSON
        # ====================================================

        for ppe in ppe_detections:

            best_person = None
            best_score = 0.0

            for person in persons:

                score = calculate_association_score(
                    person["box"],
                    ppe["box"],
                    ppe["type"]
                )

                if score > best_score:

                    best_score = score
                    best_person = person

            if (
                best_person is not None
                and
                best_score >= ASSOCIATION_THRESHOLD
            ):

                ppe_type = ppe["type"]

                current = (
                    best_person[
                        "current_ppe"
                    ][
                        ppe_type
                    ]
                )

                # ------------------------------------------------
                # NO takes priority over YES within this frame.
                # ------------------------------------------------

                if ppe["status"] == "NO":

                    best_person[
                        "current_ppe"
                    ][
                        ppe_type
                    ] = "NO"

                elif current != "NO":

                    best_person[
                        "current_ppe"
                    ][
                        ppe_type
                    ] = "YES"

        # ====================================================
        # UPDATE PPE MEMORY
        # ====================================================

        for person in persons:

            person_id = person["id"]

            if person_id is None:

                continue

            for ppe_type in PPE_TYPES:

                observation = (
                    person[
                        "current_ppe"
                    ][
                        ppe_type
                    ]
                )

                memory = person_memory[
                    person_id
                ][
                    ppe_type
                ]

                update_ppe_memory(
                    memory,
                    observation,
                    current_time
                )

        # ====================================================
        # FINAL STATUS
        # ====================================================

        for person in persons:

            person_id = person["id"]

            # ------------------------------------------------
            # No tracking ID
            # ------------------------------------------------

            if person_id is None:

                person["stable_ppe"] = {

                    "helmet": "UNKNOWN",
                    "vest": "UNKNOWN",
                    "boots": "UNKNOWN"
                }

                person["overall"] = "UNKNOWN"

                continue

            memory = person_memory[
                person_id
            ]

            stable_ppe = {

                "helmet": memory[
                    "helmet"
                ]["status"],

                "vest": memory[
                    "vest"
                ]["status"],

                "boots": memory[
                    "boots"
                ]["status"]
            }

            person["stable_ppe"] = stable_ppe

            person["overall"] = (
                calculate_overall_status(
                    stable_ppe
                )
            )

        # ====================================================
        # DATABASE LOGGING
        # ====================================================

        for person in persons:

            person_id = person["id"]

            if person_id is None:

                continue

            ppe = person["stable_ppe"]

            overall = person["overall"]

            log_status_if_changed(
                person_id,
                ppe,
                overall,
                person_memory[
                    person_id
                ]
            )

        # ====================================================
        # REMOVE OLD MEMORY
        # ====================================================

        old_ids = []

        for (
            person_id,
            memory
        ) in person_memory.items():

            if (
                frame_number
                -
                memory["last_seen"]
                >
                PERSON_MEMORY_FRAMES
            ):

                old_ids.append(
                    person_id
                )

        for person_id in old_ids:

            del person_memory[
                person_id
            ]

        # ====================================================
        # DRAW PERSON BOXES
        # ====================================================

        for person in persons:

            x1, y1, x2, y2 = (
                person["box"]
            )

            person_id = person["id"]

            overall = person["overall"]

            # ------------------------------------------------
            # PERSON BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # ------------------------------------------------
            # PERSON ID
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"ID: {person_id}",
                (
                    x1,
                    max(
                        y1 - 10,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            # ------------------------------------------------
            # OVERALL STATUS COLOR
            # ------------------------------------------------

            if overall == "COMPLIANT":

                status_color = (
                    0,
                    255,
                    0
                )

            elif overall == "NON-COMPLIANT":

                status_color = (
                    0,
                    0,
                    255
                )

            else:

                status_color = (
                    0,
                    255,
                    255
                )

            # ------------------------------------------------
            # OVERALL STATUS
            # ------------------------------------------------

            cv2.putText(
                frame,
                overall,
                (
                    x1,
                    min(
                        y2 + 25,
                        frame.shape[0] - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_color,
                2
            )

        # ====================================================
        # DRAW PPE BOXES
        # ====================================================

        for ppe in ppe_detections:

            x1, y1, x2, y2 = (
                ppe["box"]
            )

            label = (
                f"{ppe['class_name']} "
                f"{ppe['confidence']:.2f}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        y1 - 7,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1
            )

        # ====================================================
        # STREAMLIT VIDEO
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        video_placeholder.image(
            rgb_frame,
            channels="RGB",
            width="stretch"
        )

        # ====================================================
        # STATUS TEXT
        # ====================================================

        status_placeholder.write(
            f"Frame: {frame_number} | "
            f"Workers: {len(persons)} | "
            f"PPE detections: {len(ppe_detections)}"
        )

        # ====================================================
        # WORKER STATUS TABLE
        # ====================================================

        worker_data = []

        for person in persons:

            worker_data.append({

                "Person ID": person["id"],

                "Helmet": (
                    person[
                        "stable_ppe"
                    ]["helmet"]
                ),

                "Vest": (
                    person[
                        "stable_ppe"
                    ]["vest"]
                ),

                "Boots": (
                    person[
                        "stable_ppe"
                    ]["boots"]
                ),

                "Overall": person[
                    "overall"
                ]
            })

        if worker_data:

            worker_placeholder.dataframe(
                worker_data,
                width="stretch",
                hide_index=True
            )
