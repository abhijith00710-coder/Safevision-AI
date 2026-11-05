# 🛡️ SafeVision AI
### AI-Powered Industrial PPE Monitoring and Safety Intelligence System

SafeVision AI is an AI-powered industrial safety monitoring application designed to detect Personal Protective Equipment (PPE) compliance, track people across video frames, maintain safety records, and provide intelligent answers to safety-related questions.

The system combines computer vision, database management, and Retrieval-Augmented Generation (RAG) to support workplace safety monitoring.

## 🚀 Key Features

### 1. Real-Time PPE Detection
- Uses YOLO and OpenCV for person and PPE detection.
- Detects the following PPE categories:
  - Safety Helmet
  - Safety Vest
  - Safety Boots
- Uses ByteTrack for tracking detected people across video frames.
- Assigns temporary tracking IDs to detected persons.

### 2. PPE Compliance Monitoring
- Evaluates PPE status using a rule-based compliance system.
- Classifies overall worker status as:
  - **COMPLIANT** – All monitored PPE items are detected.
  - **NON-COMPLIANT** – At least one monitored PPE item is confirmed missing.
  - **UNKNOWN** – PPE status cannot be reliably confirmed.
- Displays individual PPE status for each detected person.

### 3. Safety Dashboard
- Displays detected worker information.
- Shows total PPE events.
- Provides compliant and non-compliant worker counts.
- Displays recent PPE monitoring records.
- Retrieves monitoring information from SQLite.

### 4. AI Safety Assistant
- Uses SQL retrieval to obtain worker information from the SQLite database.
- Uses RAG to retrieve relevant industrial safety policy sections from ChromaDB.
- Uses LangChain and Ollama with Llama 3.2 to generate answers.
- Combines database information and retrieved policy context to answer safety-related questions.

## 🏗️ System Architecture

```text
             Video Input
                  |
                  v
          YOLO + OpenCV
                  |
                  v
           ByteTrack
                  |
                  v
         PPE Detection Results
                  |
                  v
       Rule-Based Compliance
                  |
                  v
          SQLite Database
                  |
          +-------+-------+
          |               |
          v               v
    SQL Retrieval     RAG Retrieval
          |               |
          |          ChromaDB
          |               |
          +-------+-------+
                  |
                  v
             Ollama LLM
             (Llama 3.2)
                  |
                  v
          AI Safety Assistant
```

## 🧰 Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Computer Vision | OpenCV |
| Object Detection | YOLO, Ultralytics |
| Object Tracking | ByteTrack |
| Database | SQLite |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | Hugging Face Sentence Transformers |
| LLM | Llama 3.2 |
| Local LLM Runtime | Ollama |
| User Interface | Streamlit |

## 📂 Project Structure

```text
SafeVision-AI/
│
├── knowledge_base/
│   └── safety_policy.txt
│
├── src/
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── app_backup.py
│   │   └── live_monitor.py
│   │
│   ├── database/
│   │   ├── database.py
│   │   ├── logger.py
│   │   ├── queries.py
│   │   └── sql_assistant.py
│   │
│   ├── detection/
│   │   ├── train_ppe.py
│   │   └── view_ppe_opencv.py
│   │
│   └── rag/
│       ├── ingest.py
│       ├── llm_test.py
│       ├── rag_assistant.py
│       ├── retriever.py
│       └── sql_rag_assistant.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.11
- Git
- Ollama
- A compatible camera or configured video source
- Trained YOLO model weights

### 1. Clone the Repository

```bash
git clone https://github.com/abhijith00710-coder/Safevision-AI.git
```

```bash
cd Safevision-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install ultralytics opencv-python numpy Pillow streamlit langchain langchain-community langchain-huggingface langchain-ollama langchain-chroma chromadb sentence-transformers
```

### 4. Set Up Ollama

Install Ollama and download the Llama 3.2 model:

```bash
ollama pull llama3.2
```

Ensure Ollama is running before using the Safety Assistant.

### 5. Prepare the RAG Knowledge Base

Run the ingestion script to process the safety policy and create the ChromaDB vector store:

```bash
python -m src.rag.ingest
```

### 6. Configure the YOLO Model

Place your trained YOLO model weights in the location configured in the detection scripts.

Configure the appropriate camera or video source before starting live monitoring.

### 7. Run the Application

```bash
python -m streamlit run src/dashboard/app.py
```

The Streamlit application will open in your browser.

## 🖥️ Application Modules

The application contains three main modules:

1. **Live Monitor** – Displays live PPE detection and tracking.
2. **Safety Dashboard** – Displays worker compliance information and PPE event records.
3. **Safety Assistant** – Answers questions using SQL database retrieval and safety policy RAG.

## ⚠️ Current Limitations

- Active PPE compliance monitoring currently covers helmets, safety vests, and safety boots.
- Gloves and goggles are not currently part of the active application logic.
- Tracking IDs are temporary and do not represent permanent worker identities.
- Entry and exit counting is not implemented.
- Detection performance depends on camera quality, lighting, viewing angle, and model performance.
- The trained model weights, local database, and ChromaDB data are not included in this repository.

## 🎯 Project Objective

The objective of SafeVision AI is to demonstrate how computer vision, database systems, and Generative AI can work together to support industrial safety monitoring.

The project combines automated PPE detection, rule-based compliance evaluation, structured data retrieval, and policy-based question answering in one application.

---

**Developed as an AI and Computer Vision project.**
