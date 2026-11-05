from pathlib import Path
import re


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_PATH = (
    PROJECT_ROOT
    / "knowledge_base"
    / "safety_policy.txt"
)


# ============================================================
# POLICY SECTION MAP
# ============================================================

SECTION_MAP = {

    # --------------------------------------------------------
    # Individual PPE
    # --------------------------------------------------------

    "helmet": [
        "2. SAFETY HELMET",
        "11. MISSING PPE",
        "12. SAFETY RESPONSE",
    ],

    "vest": [
        "3. SAFETY VEST",
        "11. MISSING PPE",
        "12. SAFETY RESPONSE",
    ],

    "boots": [
        "4. SAFETY BOOTS",
        "11. MISSING PPE",
        "12. SAFETY RESPONSE",
    ],

    # --------------------------------------------------------
    # PPE status
    # --------------------------------------------------------

    "unknown": [
        "5. PPE STATUS VALUES",
        "7. COMPUTER VISION LIMITATIONS",
        "8. DETECTION AND CONFIRMATION",
    ],

    "compliance": [
        "6. OVERALL PPE COMPLIANCE",
    ],

    "non_compliant_workers": [
        "10. NON-COMPLIANT WORKERS",
        "6. OVERALL PPE COMPLIANCE",
    ],

    "missing_ppe": [
        "11. MISSING PPE",
        "12. SAFETY RESPONSE",
    ],

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    "detection_problem": [
        "7. COMPUTER VISION LIMITATIONS",
        "8. DETECTION AND CONFIRMATION",
    ],

    # --------------------------------------------------------
    # Other PPE
    # --------------------------------------------------------

    "gloves": [
        "15. GLOVES AND GOGGLES",
    ],

    "goggles": [
        "15. GLOVES AND GOGGLES",
    ],

    # --------------------------------------------------------
    # Entry / Exit
    # --------------------------------------------------------

    "entry_exit": [
        "14. ENTRY AND EXIT",
    ],

    # --------------------------------------------------------
    # General PPE
    # --------------------------------------------------------

    "general_ppe": [
        "2. SAFETY HELMET",
        "3. SAFETY VEST",
        "4. SAFETY BOOTS",
        "6. OVERALL PPE COMPLIANCE",
    ],

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    "database": [
        "9. DATABASE INFORMATION",
    ],

    # --------------------------------------------------------
    # Combined person status + policy
    #
    # This is the important addition.
    # --------------------------------------------------------

    "person_status_policy": [
        "5. PPE STATUS VALUES",
        "6. OVERALL PPE COMPLIANCE",
        "7. COMPUTER VISION LIMITATIONS",
    ],
}


# ============================================================
# LOAD POLICY
# ============================================================

def load_policy():
    """
    Load the safety policy text file.
    """

    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"Safety policy not found:\n{POLICY_PATH}"
        )

    return POLICY_PATH.read_text(
        encoding="utf-8"
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for intent detection.

    Converts:
        "What's the PPE status?"
    into a simpler searchable form.
    """

    text = text.lower()

    # Handle common contractions before removing punctuation.
    replacements = {
        "wasn't": "was not",
        "weren't": "were not",
        "isn't": "is not",
        "aren't": "are not",
        "doesn't": "does not",
        "don't": "do not",
        "didn't": "did not",
        "can't": "cannot",
        "couldn't": "could not",
        "shouldn't": "should not",
        "what's": "what is",
        "that's": "that is",
        "it's": "it is",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove punctuation.
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Remove repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PARSE POLICY SECTIONS
# ============================================================

def get_policy_sections():
    """
    Split the safety policy into numbered sections.

    Example:

        2. SAFETY HELMET
        ...

        3. SAFETY VEST
        ...

    becomes separate section records.
    """

    text = load_policy()

    pattern = r"(?m)^(\d+)\.\s+([^\n]+)"

    matches = list(
        re.finditer(
            pattern,
            text
        )
    )

    sections = {}

    for i, match in enumerate(matches):

        section_number = match.group(1)

        section_name = (
            match.group(2)
            .strip()
        )

        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end].strip()

        canonical_name = (
            f"{section_number}. {section_name}"
        )

        sections[
            normalize_text(canonical_name)
        ] = {
            "number": section_number,
            "title": section_name,
            "canonical": canonical_name,
            "content": content,
        }

    return sections


# ============================================================
# FIND SECTION
# ============================================================

def find_section(
    sections,
    section_name
):
    """
    Find a policy section by its numbered title.
    """

    target = normalize_text(
        section_name
    )

    # Exact match
    if target in sections:
        return sections[target]

    # Prefix match
    for key, section in sections.items():

        if key.startswith(target):
            return section

    return None


# ============================================================
# DETECT COMBINED PERSON + POLICY QUESTION
# ============================================================

def is_person_status_policy_question(q):
    """
    Detect questions that ask for:

        database worker status
        +
        interpretation according to safety policy

    Example:

        What is the PPE status of Person 130,
        and what does the safety policy say about this status?
    """

    # Person / worker reference
    has_person_reference = bool(
        re.search(
            r"\b(person|worker)\s+\d+\b",
            q
        )
    )

    # Status-related language
    status_phrases = [
        "ppe status",
        "status of",
        "worker status",
        "person status",
        "overall status",
        "ppe state",
        "status",
    ]

    has_status_reference = any(
        phrase in q
        for phrase in status_phrases
    )

    # Policy-related language
    policy_phrases = [
        "safety policy",
        "policy say",
        "policy says",
        "according to the policy",
        "according to safety policy",
        "what does the policy",
        "what does this status mean",
        "what does that status mean",
        "what does unknown mean",
        "what does compliant mean",
        "what does non compliant mean",
        "what does non-compliant mean",
        "is this compliant",
        "is the worker compliant",
        "does this mean compliant",
    ]

    has_policy_reference = any(
        phrase in q
        for phrase in policy_phrases
    )

    return (
        has_person_reference
        and has_status_reference
        and has_policy_reference
    )


# ============================================================
# DETECT INTENT
# ============================================================

def detect_intent(question):

    q = normalize_text(question)

    # --------------------------------------------------------
    # 1. Combined person status + policy
    #
    # IMPORTANT:
    # This must be checked BEFORE generic database routing.
    # --------------------------------------------------------

    if is_person_status_policy_question(q):
        return "person_status_policy"

    # --------------------------------------------------------
    # 2. Entry / Exit
    # --------------------------------------------------------

    if (
        "entry" in q
        or "exit" in q
        or "entered" in q
        or "left" in q
        or "people entered" in q
    ):
        return "entry_exit"

    # --------------------------------------------------------
    # 3. Gloves
    # --------------------------------------------------------

    if (
        "glove" in q
        or "gloves" in q
    ):
        return "gloves"

    # --------------------------------------------------------
    # 4. Goggles
    # --------------------------------------------------------

    if (
        "goggle" in q
        or "goggles" in q
    ):
        return "goggles"

    # --------------------------------------------------------
    # 5. UNKNOWN / UNCERTAINTY
    # --------------------------------------------------------

    if (
        "unknown" in q
        or "uncertain" in q
        or "cannot determine" in q
        or "cannot be determined" in q
        or "insufficient evidence" in q
        or "not enough evidence" in q
        or "what does unknown mean" in q
    ):
        return "unknown"

    # --------------------------------------------------------
    # 6. Detection problems
    # --------------------------------------------------------

    detection_phrases = [

        "why was not detected",
        "why was the helmet not detected",
        "why was the vest not detected",
        "why were the boots not detected",

        "why wasnt detected",
        "why wasnt the helmet detected",
        "why wasnt the vest detected",
        "why wasnt the boots detected",

        "not detected",
        "was not detected",
        "wasnt detected",

        "detection problem",
        "detection issue",
        "detection failure",

        "why did the system miss",
        "why did it miss",
        "why was it missed",

        "why did not detect",
        "why didnt the system detect",

        "why did the detector fail",
    ]

    if any(
        phrase in q
        for phrase in detection_phrases
    ):
        return "detection_problem"

    # --------------------------------------------------------
    # 7. Non-compliant workers
    # --------------------------------------------------------

    if (
        "which workers are non compliant" in q
        or "who is non compliant" in q
        or "workers are non compliant" in q
        or "who are non compliant" in q
        or "which workers are non compliant" in q
    ):
        return "non_compliant_workers"

    # --------------------------------------------------------
    # 8. Overall compliance
    # --------------------------------------------------------

    if (
        "when is a worker considered compliant" in q
        or "when is a worker compliant" in q
        or "what makes a worker compliant" in q
        or "what is compliant" in q
        or "what does compliant mean" in q
        or "ppe compliance" in q
        or "overall compliance" in q
        or "non compliant" in q
    ):
        return "compliance"

    # --------------------------------------------------------
    # 9. Missing PPE
    # --------------------------------------------------------

    missing_phrases = [

        "does not have",
        "doesnt have",

        "does not wear",
        "doesnt wear",

        "without",

        "missing",

        "not wearing",
        "not wear",

        "no helmet",
        "no vest",
        "no boots",
    ]

    if any(
        phrase in q
        for phrase in missing_phrases
    ):
        return "missing_ppe"

    # --------------------------------------------------------
    # 10. Helmet
    # --------------------------------------------------------

    if (
        "helmet" in q
        or "hard hat" in q
        or "head protection" in q
    ):
        return "helmet"

    # --------------------------------------------------------
    # 11. Vest
    # --------------------------------------------------------

    if (
        "vest" in q
        or "reflective vest" in q
        or "high visibility" in q
    ):
        return "vest"

    # --------------------------------------------------------
    # 12. Boots
    # --------------------------------------------------------

    if (
        "boots" in q
        or "boot" in q
        or "footwear" in q
        or "safety footwear" in q
    ):
        return "boots"

    # --------------------------------------------------------
    # 13. Database
    # --------------------------------------------------------

    if (
        "database" in q
        or "record" in q
        or "person" in q
        or "worker" in q
        or "how many" in q
        or "count" in q
        or "status of" in q
    ):
        return "database"

    # --------------------------------------------------------
    # 14. General PPE
    # --------------------------------------------------------

    if (
        "ppe" in q
        or "personal protective equipment" in q
        or "safety equipment" in q
        or "ppe requirement" in q
        or "ppe requirements" in q
    ):
        return "general_ppe"

    # --------------------------------------------------------
    # 15. Default
    # --------------------------------------------------------

    return "general_ppe"


# ============================================================
# SEARCH SAFETY POLICY
# ============================================================

def search_safety_policy(
    question,
    k=3
):
    """
    Retrieve relevant safety-policy sections.

    For a combined person-status + policy question,
    retrieve the policy sections that explain:

        UNKNOWN
        overall compliance
        computer-vision limitations
    """

    sections = get_policy_sections()

    intent = detect_intent(question)

    requested_sections = SECTION_MAP.get(
        intent,
        SECTION_MAP["general_ppe"]
    )

    results = []

    for section_name in requested_sections:

        section = find_section(
            sections,
            section_name
        )

        if section is None:
            continue

        results.append(
            {
                "title": section["canonical"],
                "content": section["content"],
                "score": 100 - (
                    len(results) * 10
                ),
                "intent": intent,
            }
        )

        if len(results) >= k:
            break

    return results


# ============================================================
# FORMAT POLICY RESULTS
# ============================================================

def format_policy_results(results):

    if not results:
        return (
            "No relevant safety policy "
            "information was found."
        )

    output = []

    for result in results:

        output.append(
            f"""
[{result['title']}]

{result['content']}
"""
        )

    return "\n".join(output)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_questions = [

        # Combined SQL + RAG
        "What is the PPE status of Person 130, and what does the safety policy say about this status?",

        "What is Person 130's status and what does UNKNOWN mean according to the policy?",

        "What is the PPE status of Person 130 and is this compliant according to the safety policy?",

        # Unknown
        "What does UNKNOWN PPE status mean?",

        # Compliance
        "When is a worker considered compliant?",

        # PPE
        "What are the requirements for safety boots?",

        # Detection
        "Why wasn't the helmet detected?",

        # Non-compliant
        "Which workers are non-compliant?",

        # Missing PPE
        "What happens if safety boots are missing?",

        # Entry / Exit
        "How many people entered without helmets?",

        # Gloves
        "Are gloves monitored?",

        # General PPE
        "What are the PPE requirements?",
    ]

    print()
    print("=" * 70)
    print("SafeVision AI - RAG Retriever Test")
    print("=" * 70)

    for question in test_questions:

        print()
        print("=" * 70)
        print("QUESTION:")
        print(question)
        print("=" * 70)

        intent = detect_intent(
            question
        )

        print()
        print("DETECTED INTENT:")
        print(intent)

        results = search_safety_policy(
            question,
            k=3
        )

        print()
        print("RETRIEVED POLICY SECTIONS:")

        if not results:
            print(
                "No relevant policy sections found."
            )

        for result in results:

            print()
            print(
                f"[{result['score']}] "
                f"{result['title']}"
            )

            print(
                result["content"]
            )