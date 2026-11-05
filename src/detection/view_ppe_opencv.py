import cv2
from ultralytics import YOLO
from collections import defaultdict
import sys
from pathlib import Path

# Allow access to the project's database package when this script
# is executed directly from src/detection.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.logger import log_ppe_event


# ============================================================
# 1. LOAD MODEL
# ============================================================

model = YOLO("runs/detect/train/weights/best.pt")

print("Model loaded successfully.")
print("Classes:", model.names)


# ============================================================
# 2. PPE CLASSES
# ============================================================

POSITIVE_CLASSES = {
    "helmet": "helmet",
    "vest": "vest",
    "boots": "boots",
}

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
# 3. DETECTION SETTINGS
# ============================================================

# Lower confidence helps small objects such as gloves
DETECTION_CONFIDENCE = 0.40

# Minimum score for associating PPE with a person
ASSOCIATION_THRESHOLD = 0.40


# ============================================================
# 4. MEMORY SETTINGS
# ============================================================

# Number of consecutive positive/negative observations
# required before changing a confirmed status.

CONFIRM_FRAMES = 3


# How long to remember a person after the tracker temporarily
# loses the person.

PERSON_MEMORY_FRAMES = 180


# ============================================================
# 5. PERSON PPE MEMORY
# ============================================================

#
# Structure:
#
# person_memory[person_id] = {
#
#     "last_seen": frame_number,
#
#     "helmet": {
#         "status": "YES",
#         "yes_count": 5,
#         "no_count": 0
#     },
#
#     ...
# }
#
# ============================================================

person_memory = {}


# ============================================================
# 6. FRAME COUNTER
# ============================================================

frame_number = 0


# ============================================================
# DATABASE LOGGING MEMORY
# ============================================================

# Stores the last PPE state logged for each tracked person.
# A new database record is created only when the person's
# confirmed PPE state changes.
last_logged_status = {}


# Camera identifier used in the database.
CAMERA_ID = "CAM_01"


# ============================================================
# 7. BODY REGION FUNCTION
# ============================================================

def get_body_region(person_box, ppe_box):

    px1, py1, px2, py2 = person_box

    x1, y1, x2, y2 = ppe_box

    person_width = px2 - px1
    person_height = py2 - py1

    if person_width <= 0 or person_height <= 0:
        return None, 0.0


    # PPE center

    ppe_center_x = (x1 + x2) / 2
    ppe_center_y = (y1 + y2) / 2


    # Normalized position

    nx = (ppe_center_x - px1) / person_width
    ny = (ppe_center_y - py1) / person_height


    # Allow a little area outside person box

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


    # ========================================================
    # HEAD
    # ========================================================

    if 0.00 <= ny <= 0.30:

        distance = abs(nx - 0.50)

        score = 1.0 - min(distance / 0.50, 1.0)

        return "head", score


    # ========================================================
    # TORSO
    # ========================================================

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


    # ========================================================
    # HANDS
    # ========================================================

    if 0.35 <= ny <= 0.90:

        left_distance = abs(nx - 0.10)

        right_distance = abs(nx - 0.90)

        side_distance = min(
            left_distance,
            right_distance
        )

        score = 1.0 - min(
            side_distance / 0.50,
            1.0
        )

        return "hands", score


    # ========================================================
    # FEET
    # ========================================================

    if 0.65 <= ny <= 1.15:

        vertical_score = 1.0 - min(
            abs(ny - 0.90) / 0.40,
            1.0
        )

        score = max(
            0.0,
            vertical_score
        )

        return "feet", score


    return None, 0.0


# ============================================================
# 8. PPE → PERSON ASSOCIATION SCORE
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
# 9. CREATE NEW PERSON MEMORY
# ============================================================

def create_person_memory():

    return {

        "last_seen": frame_number,

        "helmet": {
            "status": "UNKNOWN",
            "yes_count": 0,
            "no_count": 0
        },

        "vest": {
            "status": "UNKNOWN",
            "yes_count": 0,
            "no_count": 0
        },


        "boots": {
            "status": "UNKNOWN",
            "yes_count": 0,
            "no_count": 0
        }
    }


# ============================================================
# 10. UPDATE PPE MEMORY
# ============================================================

def update_ppe_memory(
    person_id,
    ppe_type,
    observation
):

    if person_id not in person_memory:

        person_memory[person_id] = (
            create_person_memory()
        )


    memory = person_memory[
        person_id
    ][
        ppe_type
    ]


    # ========================================================
    # POSITIVE OBSERVATION
    # ========================================================

    if observation == "YES":

        memory["yes_count"] += 1

        memory["no_count"] = 0


        # Confirm YES after repeated evidence

        if memory["yes_count"] >= CONFIRM_FRAMES:

            memory["status"] = "YES"


    # ========================================================
    # NEGATIVE OBSERVATION
    # ========================================================

    elif observation == "NO":

        memory["no_count"] += 1

        memory["yes_count"] = 0


        # Confirm NO after repeated evidence

        if memory["no_count"] >= CONFIRM_FRAMES:

            memory["status"] = "NO"


    # ========================================================
    # UNKNOWN / NOT DETECTED
    # ========================================================

    #
    # IMPORTANT:
    #
    # We DO NOT reset the previous confirmed status.
    #
    # If helmet was confirmed YES and becomes hidden,
    # it remains YES.
    #
    # ========================================================

    return memory["status"]


# ============================================================
# 11. OVERALL STATUS
# ============================================================

def calculate_overall_status(ppe):

    values = [
        ppe["helmet"],
        ppe["vest"],
        ppe["boots"]
    ]


    # Explicit violation

    if "NO" in values:

        return "NON-COMPLIANT"


    # Everything confirmed

    if all(
        value == "YES"
        for value in values
    ):

        return "COMPLIANT"


    # Some PPE still unknown

    return "UNKNOWN"


# ============================================================
# 12. OPEN VIDEO
# ============================================================

video_path = "data/videos/CAM_01/test_.mp4"

cap = cv2.VideoCapture(video_path)


if not cap.isOpened():

    print("ERROR: Could not open video.")

    exit()


print("Video opened successfully.")


# ============================================================
# 13. MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print("Video finished.")

        break


    frame_number += 1


    # ========================================================
    # 14. YOLO TRACKING
    # ========================================================

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


    # ========================================================
    # 15. READ DETECTIONS
    # ========================================================

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


        # ====================================================
        # 16. SEPARATE PERSON AND PPE
        # ====================================================

        for box, class_id, confidence, track_id in zip(
            boxes,
            class_ids,
            confidences,
            track_ids
        ):

            x1, y1, x2, y2 = box.astype(int)

            class_name = model.names[
                class_id
            ].lower()

            # ============================================================
            # DEBUG RAW YOLO PPE DETECTIONS
            # ============================================================

            if class_name in {
                "helmet",
                "no_helmet",
                "vest",
                "boots",
                "no_boots"
            }:
                print(
                    f"[RAW YOLO] "
                    f"Frame={frame_number} | "
                    f"Class={class_name} | "
                    f"Confidence={confidence:.2f}"
                )


            # -----------------------------------------------
            # PERSON
            # -----------------------------------------------

            if class_name == "person":

                persons.append({

                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),

                    "id": track_id,

                    "confidence": confidence,

                    "current_ppe": {
                        "helmet": None,
                        "vest": None,
                        "gloves": None,
                        "boots": None
                    }
                })


            # -----------------------------------------------
            # POSITIVE PPE
            # -----------------------------------------------

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


            # -----------------------------------------------
            # NEGATIVE PPE
            # -----------------------------------------------

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


    # ========================================================
    # 17. UPDATE LAST-SEEN TIME
    # ========================================================

    for person in persons:

        person_id = person["id"]


        if person_id is not None:

            if person_id not in person_memory:

                person_memory[person_id] = (
                    create_person_memory()
                )


            person_memory[
                person_id
            ][
                "last_seen"
            ] = frame_number


    # ========================================================
    # 18. ASSOCIATE PPE WITH PERSON
    # ========================================================

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


        # ----------------------------------------------------
        # Assign PPE
        # ----------------------------------------------------

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


            # NO takes priority

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


    # ========================================================
    # 19. UPDATE MEMORY ONLY WHEN WE HAVE EVIDENCE
    # ========================================================

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


            # IMPORTANT:
            #
            # If observation is None,
            # we DO NOT modify memory.
            #
            # This preserves the previous confirmed result.
            #

            if observation is not None:

                update_ppe_memory(
                    person_id,
                    ppe_type,
                    observation
                )


    # ========================================================
    # 20. GET FINAL STATUS FROM MEMORY
    # ========================================================

    for person in persons:

        person_id = person["id"]


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

            "helmet": memory["helmet"]["status"],

            "vest": memory["vest"]["status"],


            "boots": memory["boots"]["status"]
        }


        person["stable_ppe"] = stable_ppe


        person["overall"] = (
            calculate_overall_status(
                stable_ppe
            )
        )


    # ========================================================
    # 21. LOG PPE EVENT TO DATABASE
    # ========================================================

    for person in persons:

        person_id = person["id"]

        # Only tracked people can be reliably logged.
        if person_id is None:
            continue

        ppe = person["stable_ppe"]
        overall = person["overall"]

        # Convert the current confirmed PPE state into a tuple.
        current_status = (
            ppe["helmet"],
            ppe["vest"],
            ppe["boots"],
            overall
        )

        # Get the previously logged state for this person.
        previous_status = last_logged_status.get(
            person_id
        )

        # Log only the first confirmed state or when
        # the confirmed state changes.
        if (
            previous_status is None
            or previous_status != current_status
        ):

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

                last_logged_status[person_id] = (
                    current_status
                )

                print(
                    f"[DATABASE] Logged PPE_CHECK | "
                    f"Person {int(person_id)} | "
                    f"Helmet={ppe['helmet']} | "
                    f"Vest={ppe['vest']} | "
                    f"Boots={ppe['boots']} | "
                    f"Overall={overall}"
                )

            except Exception as error:

                print(
                    f"[DATABASE ERROR] "
                    f"Could not log Person {int(person_id)}: "
                    f"{error}"
                )


    # ========================================================
    # 22. REMOVE OLD PERSON MEMORY
    # ========================================================

    old_ids = []


    for person_id, memory in person_memory.items():

        frames_since_seen = (
            frame_number
            -
            memory["last_seen"]
        )


        if frames_since_seen > PERSON_MEMORY_FRAMES:

            old_ids.append(person_id)


    for person_id in old_ids:

        del person_memory[person_id]


    # ========================================================
    # 22. DRAW PERSON BOXES
    # ========================================================

    for person in persons:

        x1, y1, x2, y2 = person["box"]

        person_id = person["id"]


        # Person box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # Center point

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )


        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )


        # Bottom point

        cv2.circle(
            frame,
            (center_x, y2),
            6,
            (255, 0, 0),
            -1
        )


        # ID

        cv2.putText(
            frame,
            f"ID:{person_id}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # ========================================================
    # 23. DRAW PPE BOXES
    # ========================================================

    for ppe in ppe_detections:

        x1, y1, x2, y2 = ppe["box"]


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
            (x1, max(y1 - 7, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )


    # ========================================================
    # 24. CREATE LARGE SIDE PANEL
    # ========================================================

    panel_width = 430


    panel = cv2.copyMakeBorder(
        frame,
        0,
        0,
        0,
        panel_width,
        cv2.BORDER_CONSTANT,
        value=(25, 25, 25)
    )


    panel_x = frame.shape[1] + 20


    # ========================================================
    # 25. PANEL TITLE
    # ========================================================

    cv2.putText(
        panel,
        "SAFETY MONITOR",
        (panel_x, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )


    cv2.putText(
        panel,
        f"People: {len(persons)}",
        (panel_x, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 26. PERSON STATUS
    # ========================================================

    y = 110


    for person in persons:

        person_id = person["id"]

        ppe = person["stable_ppe"]

        overall = person["overall"]


        # ----------------------------------------------------
        # PERSON
        # ----------------------------------------------------

        cv2.putText(
            panel,
            f"PERSON ID: {person_id}",
            (panel_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2
        )


        y += 27


        # ----------------------------------------------------
        # PPE
        # ----------------------------------------------------

        status_lines = [

            ("Helmet", ppe["helmet"]),

            ("Vest", ppe["vest"]),

            ("Boots", ppe["boots"])
        ]


        for name, status in status_lines:

            if status == "YES":

                text_color = (0, 255, 0)

            elif status == "NO":

                text_color = (0, 0, 255)

            else:

                text_color = (180, 180, 180)


            cv2.putText(
                panel,
                f"{name}: {status}",
                (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                text_color,
                1
            )


            y += 22


        # ----------------------------------------------------
        # OVERALL
        # ----------------------------------------------------

        if overall == "COMPLIANT":

            overall_color = (0, 255, 0)

        elif overall == "NON-COMPLIANT":

            overall_color = (0, 0, 255)

        else:

            overall_color = (0, 255, 255)


        cv2.putText(
            panel,
            f"OVERALL: {overall}",
            (panel_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            overall_color,
            2
        )


        y += 42


    # ========================================================
    # 27. DISPLAY
    # ========================================================

    cv2.namedWindow(
        "SafeVision AI - PPE Compliance",
        cv2.WINDOW_NORMAL
    )


    cv2.imshow(
        "SafeVision AI - PPE Compliance",
        panel
    )


    # ========================================================
    # 28. QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        print("Q pressed. Closing...")

        break


# ============================================================
# 29. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Program closed successfully.")