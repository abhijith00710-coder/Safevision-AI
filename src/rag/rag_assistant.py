from pathlib import Path

import ollama

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "chroma_db"
)

MODEL_NAME = "llama3.2:latest"
TOP_K = 3


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# LOAD VECTOR DATABASE
# --------------------------------------------------

print("Loading vector database...")

vector_store = Chroma(
    persist_directory=str(VECTOR_DB_PATH),
    embedding_function=embeddings
)

print("Vector database loaded.")


# --------------------------------------------------
# RETRIEVE POLICY
# --------------------------------------------------

def retrieve_policy(question, k=TOP_K):

    results = vector_store.similarity_search_with_score(
        question,
        k=k
    )

    return results


# --------------------------------------------------
# BUILD POLICY CONTEXT
# --------------------------------------------------

def build_policy_context(question, k=TOP_K):

    results = retrieve_policy(
        question,
        k=k
    )

    context_parts = []

    for index, (document, score) in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
POLICY SOURCE {index}

{document.page_content}
""".strip()
        )

    return "\n\n".join(context_parts)


# --------------------------------------------------
# BUILD RAG PROMPT
# --------------------------------------------------

def build_rag_prompt(
    question,
    k=TOP_K
):

    context = build_policy_context(
        question,
        k=k
    )

    prompt = f"""
You are SafeVision AI, a construction-site
safety information assistant.

Answer the user's question using ONLY the
safety policy context provided below.

Do not invent safety rules.

If the policy does not contain enough information
to answer the question, say that the safety policy
does not provide enough information.

Important rules:

- Do not convert UNKNOWN into YES or NO.
- Do not invent PPE requirements.
- Do not invent disciplinary actions.
- Do not independently impose penalties.
- Do not replace qualified safety personnel.
- Distinguish computer-vision observations from
  safety-policy requirements.

SAFETY POLICY CONTEXT
=====================

{context}

USER QUESTION
=============

{question}

ANSWER
======
"""

    return prompt


# --------------------------------------------------
# LLM
# --------------------------------------------------

def ask_llm(prompt):

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# --------------------------------------------------
# ANSWER POLICY QUESTION
# --------------------------------------------------

def answer_policy_question(question):

    prompt = build_rag_prompt(
        question
    )

    answer = ask_llm(
        prompt
    )

    return answer


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    print()
    print("====================================")
    print("SafeVision AI RAG Assistant")
    print("====================================")

    question = (
        "What should happen if a worker "
        "does not have a safety helmet?"
    )

    print()
    print("QUESTION:")
    print(question)

    print()
    print("RETRIEVING POLICY...")
    print("------------------------------------")

    results = retrieve_policy(
        question
    )

    for index, (document, score) in enumerate(
        results,
        start=1
    ):

        print(
            f"\nPolicy Source {index}"
        )

        print(
            f"Distance: {score:.4f}"
        )

        print(
            document.page_content
        )

    print()
    print("LLM ANSWER:")
    print("------------------------------------")

    answer = answer_policy_question(
        question
    )

    print(answer)

    print()
    print("====================================")
    print("RAG test completed.")
    print("====================================")