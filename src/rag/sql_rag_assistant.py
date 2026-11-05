from langchain_ollama import OllamaLLM

from src.database.sql_assistant import answer_database_question
from src.rag.retriever import search_safety_policy, format_policy_results


# ============================================================
# LLM
# ============================================================

llm = OllamaLLM(
    model="llama3.2:latest"
)


# ============================================================
# DATABASE
# ============================================================

def get_database_information(question):
    """
    Retrieve information from the SafeVision SQLite database.
    """

    try:
        result = answer_database_question(question)

        if result is None:
            return "No database information was found."

        return str(result)

    except Exception as e:
        return f"Database retrieval error: {e}"


# ============================================================
# SAFETY POLICY / RAG
# ============================================================

def get_policy_information(question):
    """
    Retrieve relevant safety-policy sections using the
    strict intent-based retriever.
    """

    try:
        results = search_safety_policy(question, k=3)

        if not results:
            return "No relevant safety policy information was found."

        return format_policy_results(results)

    except Exception as e:
        return f"Safety policy retrieval error: {e}"


# ============================================================
# LLM ANSWER
# ============================================================

def generate_safety_answer(
    question,
    database_information,
    policy_information
):
    """
    Generate the final answer using ONLY:
    1. Database information
    2. Safety policy information
    """

    prompt = f"""
You are the SafeVision AI Safety Assistant.

Answer the user's question using ONLY the supplied
database information and safety policy information.

Do not invent:
- worker IDs
- PPE status
- counts
- safety rules
- application capabilities
- events
- entry/exit information

IMPORTANT RULES:

1. Database information is the source of truth for
   observed worker PPE status.

2. Safety policy is the source of truth for safety rules.

3. UNKNOWN does NOT mean NO.

4. UNKNOWN does NOT mean YES.

5. Only explicitly confirmed NO means that a PPE item
   is confirmed missing.

6. COMPLIANT means:
   Helmet = YES
   AND Vest = YES
   AND Boots = YES

7. NON-COMPLIANT means:
   At least one monitored PPE category is explicitly NO.

8. If Entry/Exit tracking is not implemented, clearly
   state that the system cannot provide entry/exit counts.

9. If gloves or goggles are asked about, state that they
   are not currently part of the active application logic.

10. If the database contains no matching records, say so.
    Do not invent records.

11. If policy information is unavailable, do not pretend
    that policy information was retrieved.

12. Distinguish between:
    - observed database status
    - safety policy rule
    - system limitation

USER QUESTION:
{question}

DATABASE INFORMATION:
{database_information}

SAFETY POLICY INFORMATION:
{policy_information}

Now provide a clear and concise answer.
"""


    try:
        response = llm.invoke(prompt)
        return response

    except Exception as e:
        return f"LLM generation error: {e}"


# ============================================================
# COMPLETE SQL + RAG + LLM PIPELINE
# ============================================================

def answer_safety_question(question):

    print()
    print("Retrieving database information...")
    database_information = get_database_information(question)

    print("Retrieving safety policy...")
    policy_information = get_policy_information(question)

    print("Generating final answer...")

    answer = generate_safety_answer(
        question,
        database_information,
        policy_information
    )

    return {
        "question": question,
        "database": database_information,
        "policy": policy_information,
        "answer": answer
    }


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("SafeVision AI")
    print("SQL + RAG + LLM Safety Assistant")
    print("======================================")

    question = input("\nAsk a safety question: ")

    result = answer_safety_question(question)

    print()
    print("======================================")
    print("DATABASE INFORMATION")
    print("======================================")
    print(result["database"])

    print()
    print("======================================")
    print("SAFETY POLICY INFORMATION")
    print("======================================")
    print(result["policy"])

    print()
    print("======================================")
    print("FINAL ANSWER")
    print("======================================")
    print(result["answer"])