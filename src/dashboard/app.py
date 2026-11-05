import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SafeVision AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ SafeVision AI")
st.caption("Industrial PPE Monitoring & Safety Intelligence")

st.divider()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📹 Live Monitor",
    "📊 Safety Dashboard",
    "🤖 Safety Assistant"
])


# ============================================================
# TAB 1 — LIVE MONITOR
# ============================================================

with tab1:

    from src.dashboard.live_monitor import show_live_monitor

    show_live_monitor()


# ============================================================
# TAB 2 — SAFETY DASHBOARD
# ============================================================

with tab2:
    @st.fragment(run_every="1s")
    def show_dashboard():

        st.header("📊 Safety Dashboard")

        from src.database.queries import (
            get_total_event_count,
            get_unique_worker_count,
            get_recent_ppe_events,
            get_latest_worker_status,
        )

        latest_workers = get_latest_worker_status()

        compliant_workers = 0
        non_compliant_workers = 0
        unknown_workers = 0

        for worker in latest_workers:
            status = worker["overall_status"]

            if status == "COMPLIANT":
                compliant_workers += 1
            elif status == "NON-COMPLIANT":
                non_compliant_workers += 1
            else:
                unknown_workers += 1

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Workers Detected",
                get_unique_worker_count()
            )

        with col2:
            st.metric(
                "PPE Events",
                get_total_event_count()
            )

        with col3:
            st.metric(
                "Compliant",
                compliant_workers
            )

        with col4:
            st.metric(
                "Non-Compliant",
                non_compliant_workers
            )

        with col5:
            st.metric(
                "Unknown",
                unknown_workers
            )

        st.divider()

        st.subheader("👷 Current Worker Status")

        if latest_workers:
            for worker in latest_workers:
                st.write(
                    f"**Person {worker['person_id']}** | "
                    f"Helmet: {worker['helmet']} | "
                    f"Vest: {worker['vest']} | "
                    f"Boots: {worker['boots']} | "
                    f"Overall: **{worker['overall_status']}**"
                )
        else:
            st.info("No worker records available.")

        st.divider()

        st.subheader("📋 Recent PPE Events")

        events = get_recent_ppe_events(20)

        if events:
            st.dataframe(
                events,
                width="stretch"
            )
        else:
            st.info("No PPE events available.")

    show_dashboard()


# ============================================================
# TAB 3 — SAFETY ASSISTANT
# ============================================================

with tab3:
    st.header("🤖 SafeVision Safety Assistant")

    st.write(
        "Ask questions about observed PPE status and "
        "the SafeVision safety policy."
    )

    if "sv_chat_history" not in st.session_state:
        st.session_state.sv_chat_history = []

    # Example questions
    st.markdown("### 💡 Example Questions")

    example_questions = [
        "What is the current status of Person 148?",
        "Which workers are currently non-compliant?",
        "Is Person 148 compliant according to the safety policy?",
        "What PPE is required for compliance?",
        "What does UNKNOWN PPE status mean?",
        "Which workers are currently without a helmet?",
        "What happens if a worker is missing safety boots?",
        "Are gloves currently monitored?",
    ]

    cols = st.columns(2)

    for i, example in enumerate(example_questions):

        with cols[i % 2]:

            if st.button(
                example,
                key=f"example_question_{i}",
                width="stretch"
            ):

                st.session_state.sv_chat_history.append({
                    "role": "user",
                    "content": example
                })

                with st.spinner(
                    "Checking live database and safety policy..."
                ):

                    from src.rag.sql_rag_assistant import (
                        answer_safety_question
                    )

                    result = answer_safety_question(example)

                    answer = result["answer"]

                st.session_state.sv_chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

                st.rerun()

    st.divider()

    # Chat history
    for message in st.session_state.sv_chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Normal user question
    question = st.chat_input(
        "Ask about current PPE status or safety policy..."
    )

    if question:

        st.session_state.sv_chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Checking live database and safety policy..."
            ):

                from src.rag.sql_rag_assistant import (
                    answer_safety_question
                )

                result = answer_safety_question(question)

                answer = result["answer"]

            st.markdown(answer)

        st.session_state.sv_chat_history.append({
            "role": "assistant",
            "content": answer
        })
# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SafeVision AI | PPE Monitoring • SQLite • RAG • LLM"
)

# python -m streamlit run src\dashboard\app.py