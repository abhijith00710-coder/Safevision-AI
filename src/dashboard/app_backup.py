import streamlit as st
import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# DATABASE IMPORTS
# ============================================================

from src.database.queries import (
    get_total_event_count,
    get_unique_worker_count,
    get_recent_ppe_events,
    get_compliant_events,
    get_non_compliant_events,
    get_unknown_events,
)


# ============================================================
# SAFETY ASSISTANT
# ============================================================

from src.rag.sql_rag_assistant import (
    answer_safety_question
)


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SafeVision AI",
    page_icon="🦺",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🦺 SafeVision AI")

st.subheader(
    "Industrial PPE Safety Monitoring System"
)

st.markdown(
    """
    SafeVision AI monitors worker PPE compliance using
    computer vision and provides safety information using
    SQLite, RAG, and an LLM-based safety assistant.
    """
)

st.divider()


# ============================================================
# DATABASE METRICS
# ============================================================

total_events = get_total_event_count()

total_workers = get_unique_worker_count()

compliant_events = len(
    get_compliant_events()
)

non_compliant_events = len(
    get_non_compliant_events()
)

unknown_events = len(
    get_unknown_events()
)


# ============================================================
# METRIC CARDS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Workers Detected",
        total_workers
    )


with col2:

    st.metric(
        "PPE Events",
        total_events
    )


with col3:

    st.metric(
        "Compliant",
        compliant_events
    )


with col4:

    st.metric(
        "Non-Compliant",
        non_compliant_events
    )


with col5:

    st.metric(
        "Unknown",
        unknown_events
    )


# ============================================================
# RECENT PPE EVENTS
# ============================================================

st.divider()

st.subheader(
    "📋 Recent PPE Events"
)


events = get_recent_ppe_events(
    20
)


if events:

    table_data = []

    for event in events:

        table_data.append(
            {
                "ID": event["id"],
                "Timestamp": event["timestamp"],
                "Camera": event["camera_id"],
                "Person": event["person_id"],
                "Helmet": event["helmet"],
                "Vest": event["vest"],
                "Boots": event["boots"],
                "Overall Status": event["overall_status"],
                "Event": event["event_type"],
            }
        )


    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True
    )


else:

    st.info(
        "No PPE events have been recorded yet."
    )


# ============================================================
# PPE STATUS SUMMARY
# ============================================================

st.divider()

st.subheader(
    "📊 PPE Status Summary"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        "### ✅ Compliant"
    )

    st.write(
        f"{compliant_events} PPE events"
    )


with col2:

    st.markdown(
        "### ⚠️ Non-Compliant"
    )

    st.write(
        f"{non_compliant_events} PPE events"
    )


with col3:

    st.markdown(
        "### ❓ Unknown"
    )

    st.write(
        f"{unknown_events} PPE events"
    )


# ============================================================
# SAFETY ASSISTANT
# ============================================================

st.divider()

st.header(
    "🤖 SafeVision Safety Assistant"
)

st.write(
    """
    Ask questions about worker PPE status, database
    records, or the safety policy.
    """
)


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Ask a safety question:",
    placeholder=(
        "Example: What is the PPE status of Person 130?"
    )
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "Ask Safety Assistant",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Analyzing database and safety policy..."
        ):

            result = answer_safety_question(
                question
            )


        # ====================================================
        # FINAL ANSWER
        # ====================================================

        st.subheader(
            "🧠 Assistant Answer"
        )

        st.success(
            result["answer"]
        )


        # ====================================================
        # OPTIONAL DETAILS
        # ====================================================

        with st.expander(
            "🔍 View database information"
        ):

            st.write(
                result["database"]
            )


        with st.expander(
            "📚 View retrieved safety policy"
        ):

            st.write(
                result["policy"]
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SafeVision AI | "
    "Computer Vision + ByteTrack + SQLite + RAG + LLM"
)