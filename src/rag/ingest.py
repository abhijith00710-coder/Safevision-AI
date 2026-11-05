from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_PATH = (
    PROJECT_ROOT
    / "knowledge_base"
    / "safety_policy.txt"
)

VECTOR_DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "chroma_db"
)


print("Loading safety policy...")

loader = TextLoader(
    str(POLICY_PATH),
    encoding="utf-8"
)

documents = loader.load()

print(
    f"Loaded {len(documents)} document(s)."
)


# ============================================================
# SPLIT POLICY INTO INDIVIDUAL SECTIONS
# ============================================================

text = documents[0].page_content

sections = []

current_section = []

for line in text.splitlines():

    # A new section starts with:
    # 1. PURPOSE
    # 2. REQUIRED PPE
    # 3. SAFETY HELMET
    # etc.
    if (
        line.strip()
        and len(line.strip()) > 2
        and line.strip()[0].isdigit()
        and "." in line.strip()[:3]
    ):

        if current_section:
            sections.append(
                "\n".join(current_section).strip()
            )

        current_section = [line]

    else:
        current_section.append(line)


if current_section:
    sections.append(
        "\n".join(current_section).strip()
    )


print(
    f"Created {len(sections)} policy sections."
)


# ============================================================
# CREATE LANGCHAIN DOCUMENTS
# ============================================================

from langchain_core.documents import Document

section_documents = []

for section in sections:

    if section.strip():

        section_documents.append(
            Document(
                page_content=section,
                metadata={
                    "source": str(POLICY_PATH),
                    "document_type": "safety_policy"
                }
            )
        )


# ============================================================
# EMBEDDINGS
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

print("Creating vector database...")

vector_store = Chroma.from_documents(
    documents=section_documents,
    embedding=embeddings,
    persist_directory=str(VECTOR_DB_PATH)
)

print("Vector database created successfully.")

print()
print("====================================")
print("SafeVision AI RAG Ingestion")
print("====================================")
print(f"Policy: {POLICY_PATH}")
print(f"Sections: {len(section_documents)}")
print(f"Vector DB: {VECTOR_DB_PATH}")
print("====================================")