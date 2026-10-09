# 🎓 Autonomous MirAI Student Policy Advisor

> **Course:** Generative AI & LangChain Architecture  
> **Format:** Full-Stack Solo Capstone Project  
> **Primary Dataset:** `Mirai_SoT_Policy_Handbook_2026.pdf`

---

## 📌 1. Project Overview

University administrations regularly process repetitive inquiries regarding attendance debarment thresholds, grading rubrics, medical leave workflows, club founding policies, and disciplinary codes of conduct.

The **Autonomous MirAI Student Policy Advisor** is a production-grade **Retrieval-Augmented Generation (RAG)** system designed to autonomously advise students while adhering to institutional guardrails. The system guarantees **zero hallucination**: all responses are strictly confined to the verified factual clauses contained in the official *MirAI School of Technology Policy Handbook 2026*.

---

## 🏗️ 2. Architecture & Technical Design

```
+-----------------------------------------------------------------------------------+
|                                STUDENT INTERACTION LAYER                         |
|   +---------------------------------------------------------------------------+   |
|   |                  Streamlit UI (frontend.py) - Port 8501                   |   |
|   |   - Interactive Policy Chat         - 1-Click Certification Audit Prompts  |   |
|   |   - PDF Ingestion Uploader          - Expandable Source Citations & Pages |   |
|   +---------------------------------------------------------------------------+   |
+------------------------------------------|----------------------------------------+
                                           | HTTP REST (FastAPI)
+------------------------------------------v----------------------------------------+
|                               FASTAPI BACKEND (backend.py)                        |
|   +--------------------------+           +------------------------------------+   |
|   |  POST /ingest (Multipart)|           |  POST /chat (JSON Payload)         |   |
|   +-------------|------------+           +------------------|-----------------+   |
+-----------------|-------------------------------------------|---------------------+
                  |                                           |
+-----------------v------------------+     +------------------v---------------------+
|         DATA PIPELINE              |     |          RAG LCEL REASONING            |
| 1. PyPDFLoader                     |     | 1. MultiQueryRetriever (Query Expansion|
| 2. RecursiveCharacterTextSplitter  |     |    bridges student & formal jargon)    |
|    (chunk_size=1000, overlap=200)  |     | 2. Context Aggregator with Citations   |
| 3. GoogleGenerativeAIEmbeddings    |     | 3. Guardrail Prompt (Zero-Hallucination|
|    (text-embedding-004)            |     | 4. ChatGoogleGenerativeAI              |
| 4. Local ChromaDB Store            |     |    (gemini-1.5-flash, temp=0.0)        |
+------------------------------------+     +----------------------------------------+
```

### Key Technical Components:
1. **Document Loader & Semantic Chunking**:
   - `PyPDFLoader` to parse structural policy documents.
   - `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`) to preserve context across multi-tiered institutional clauses.
2. **Vector Database & Embeddings**:
   - `GoogleGenerativeAIEmbeddings` (`models/text-embedding-004`) for high-fidelity semantic embeddings.
   - `ChromaDB` (`./chroma_db`) with local persistence and metadata tagging.
3. **Advanced Retrieval Architecture**:
   - `MultiQueryRetriever` to rewrite user queries into multiple semantic angles, bridging casual student phrasing (e.g., *"bunking"*, *"fine"*, *"sick"*) with formal handbook terminology (e.g., *"attendance debarment"*, *"7-day medical submission rule"*).
4. **Guarded Generation**:
   - `ChatGoogleGenerativeAI(temperature=0.0)` for deterministic responses.
   - Explicit system prompt instructions preventing extrapolation, requiring refusal if unstated, and barring fabricated monetary fines.

---

## 🚀 3. Standard Operating Procedures (SOP)

### 3.1 Prerequisites
- **Python 3.9+** (Python 3.10+ recommended)
- **Google Gemini API Key** ([Get your API Key from Google AI Studio](https://aistudio.google.com/))

### 3.2 Installation & Environment Setup

```bash
# 1. Clone the repository
git clone <your-repo-link>
cd gen-ai2

# 2. Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# 3. Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
```

Edit your `.env` file and insert your Google Gemini API key:
```ini
GOOGLE_API_KEY=AIzaSy...your_actual_key_here
GEMINI_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=models/text-embedding-004
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
PORT=8000
BACKEND_URL=http://localhost:8000
```

---

### 3.3 Running the FastAPI Backend Server

```bash
# Activate virtual environment
source venv/bin/activate

# Launch the FastAPI backend
python backend.py
# Or with uvicorn directly:
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```
The backend will automatically index `Mirai_SoT_Policy_Handbook_2026.pdf` on startup if the database is not already initialized.
- **API Docs (Swagger):** `http://localhost:8000/docs`
- **Health Endpoint:** `http://localhost:8000/health`

---

### 3.4 Running the Streamlit Frontend

In a separate terminal:
```bash
# Activate virtual environment
source venv/bin/activate

# Launch the Streamlit application
streamlit run frontend.py
```
Open your browser at `http://localhost:8501`.

---

## 📡 4. REST API Reference

### 1. `POST /ingest`
Accepts a PDF document via multipart form upload, splits text, creates embeddings, and updates ChromaDB.

**Request:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@Mirai_SoT_Policy_Handbook_2026.pdf"
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully ingested and indexed 16 chunks from 'Mirai_SoT_Policy_Handbook_2026.pdf'.",
  "pdf_source": "Mirai_SoT_Policy_Handbook_2026.pdf",
  "pages_loaded": 16,
  "chunks_created": 16,
  "persist_directory": "./chroma_db"
}
```

---

### 2. `POST /chat`
Executes the guarded MultiQuery RAG chain and returns the answer with cited source pages.

**Request:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "I have 72% attendance. How many attendance marks will I get?",
    "k": 5
  }'
```

**Response:**
```json
{
  "question": "I have 72% attendance. How many attendance marks will I get?",
  "answer": "Based on the Standard Attendance Evaluation policy in the MirAI Student Policy Handbook, attendance carries a 10% weightage. Since your attendance is 72% (which falls into the 60% – 74.99% tier), you will receive 4 marks out of 10. Note that students must maintain at least 75% attendance overall to avoid examination debarment.",
  "sources": [
    {
      "source_id": 1,
      "page": 3,
      "content": "Standard Attendance Evaluation: Attendance carries a 10% weightage across all courses. Marks are awarded based on the following tier system: 90% and above -> 10 marks, 80% – 89.99% -> 8 marks, 75% – 79.99% -> 6 marks, 60% – 74.99% -> 4 marks, Below 60% -> 0 marks...",
      "metadata": {"source_file": "Mirai_SoT_Policy_Handbook_2026.pdf", "page_number": 3}
    }
  ],
  "status": "success"
}
```

---

## 🧪 5. Certification Audit (Test Suite)

The system is calibrated and verified against the 4 core institutional audit queries:

| Test ID | Category | Audit Question | Expected Outcome | System Compliance |
|---|---|---|---|:---:|
| **TC-01** | **Precision Verification** | *"I have 72% attendance. How many attendance marks will I get?"* | States the student receives **4 marks** (60%–74.99% tier). | ✅ Pass |
| **TC-02** | **Multi-Hop Reasoning** | *"I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?"* | Directs to **Yashaswini Ma'am** (Ratnam CM) and specifies the **7-day submission deadline** from illness/treatment. | ✅ Pass |
| **TC-03** | **Process Verification** | *"We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?"* | Explains **40% batch support** is required and proposal must go to **Faculty Coordinator first**, not Management. | ✅ Pass |
| **TC-04** | **Negative Constraint** | *"How much is the fine for smoking a cigarette on campus?"* | Confirms tobacco is prohibited and subject to **Disciplinary Committee**, but **does NOT fabricate any monetary fine**. | ✅ Pass |

---

## 📊 6. Automated Evaluation (LLM-as-a-Judge)

To run the automated benchmarking pipeline:

```bash
python evaluate.py
```

This script:
1. Executes all 4 benchmark queries through the RAG pipeline.
2. Evaluates the outputs against official context using an LLM-as-a-judge rubric (1–5 scale).
3. Logs full benchmarking metrics to [`rag_eval_scores.csv`](file:///Users/saifuddinm2311/Desktop/gen-ai2/rag_eval_scores.csv).

---

## 📂 7. Repository Structure

```
gen-ai2/
├── Mirai_SoT_Policy_Handbook_2026.pdf  # Source policy dataset
├── backend.py                          # FastAPI async REST server
├── frontend.py                         # Streamlit student UI
├── rag_pipeline.py                     # LangChain LCEL RAG core
├── evaluate.py                         # LLM-as-a-judge benchmark script
├── rag_eval_scores.csv                 # Automated benchmark audit log
├── requirements.txt                    # Project dependencies
├── .env.example                        # Template environment variables
├── .gitignore                          # Git ignore rules
└── README.md                           # Documentation & SOPs
```

---

## 🛡️ 8. Security & Ethical Guardrails

- **Zero-Hallucination System Prompt**: Restricts reasoning strictly to retrieved handbook embeddings.
- **Negative Constraint Handling**: Rejects requests to guess policy numbers or penalties not explicitly specified.
- **Data Privacy**: Local vector database isolation with persistent directory control.
