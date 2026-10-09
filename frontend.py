import os
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Page Configuration
st.set_page_config(
    page_title="MirAI Student Policy Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    .badge-status {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .badge-online {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-offline {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .policy-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }
    
    .source-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #6366F1;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.88rem;
    }
    
    .query-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
        padding: 6px 12px;
        margin: 4px;
        cursor: pointer;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health():
    """Check connection to FastAPI backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your **Autonomous MirAI Student Policy Advisor**. I can answer any institutional queries regarding attendance rules, grading frameworks, medical leaves, club formations, and disciplinary guidelines strictly based on the official **MirAI Student Policy Handbook 2026**.\n\nHow can I help you today?",
            "sources": []
        }
    ]

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1541339907198-e08756dedf3f?w=600&auto=format&fit=crop&q=80", caption="MirAI School of Technology", use_container_width=True)
    st.markdown("### 🏛️ System Control & Status")
    
    # Check backend status
    health_data = check_backend_health()
    if health_data:
        st.markdown(
            '<div class="badge-status badge-online">● Backend Connected (FastAPI)</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"**Vector Store Chunks:** `{health_data.get('document_chunk_count', 0)}`")
        st.markdown(f"**LLM Model:** `{health_data.get('llm_model', 'gemini-1.5-flash')}`")
        st.markdown(f"**Embedding Model:** `{health_data.get('embedding_model', 'text-embedding-004')}`")
    else:
        st.markdown(
            '<div class="badge-status badge-offline">● Backend Offline (Port 8000)</div>',
            unsafe_allow_html=True
        )
        st.warning("⚠️ FastAPI backend is not reachable at `http://localhost:8000`. Please start `python backend.py`.")

    st.markdown("---")
    st.markdown("### 🔑 Google Gemini API Key")
    input_key = st.text_input(
        "Enter Gemini API Key (starts with AIzaSy...)",
        type="password",
        placeholder="AIzaSy...",
        help="Paste your Google AI Studio API key here"
    )
    if st.button("💾 Save API Key", use_container_width=True):
        if input_key.strip():
            try:
                res = requests.post(f"{BACKEND_URL}/config/key", json={"api_key": input_key.strip()}, timeout=5)
                if res.status_code == 200:
                    st.success("✅ API key saved successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to save API key: {res.text}")
            except Exception as ex:
                st.error(f"❌ Error connecting to backend: {ex}")
        else:
            st.warning("Please enter a valid API key.")

    st.markdown("---")
    st.markdown("### 📄 Policy Document Ingestion")
    uploaded_file = st.file_uploader(
        "Upload MirAI Policy Handbook (PDF)",
        type=["pdf"],
        help="Upload official PDF to index into ChromaDB"
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Ingest & Index PDF", use_container_width=True):
            with st.spinner("Processing semantic chunks and vectorizing with Gemini..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=60)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Ingested {data['chunks_created']} chunks from {data['pdf_source']}!")
                        st.rerun()
                    else:
                        st.error(f"❌ Ingestion failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Ingestion request error: {str(e)}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat history cleared. How can I assist you with MirAI institutional policies today?",
                "sources": []
            }
        ]
        st.rerun()

    st.markdown("---")
    st.caption("🔒 **Strict Guardrail Active**: Zero hallucination policy. Answers are strictly confined to the MirAI Student Policy Handbook 2026.")


# Main Application Interface
st.markdown('<div class="main-title">Autonomous MirAI Student Policy Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Official AI Governance & Policy Verification System | Generative AI & LangChain Architecture</div>', unsafe_allow_html=True)

# Certification Audit Quick-Test Prompts
st.markdown("##### ⚡ Quick Certification Audit Test Queries:")
col1, col2 = st.columns(2)

prompt_to_send = None

with col1:
    if st.button("📊 1. Attendance Marks (72%)", use_container_width=True, help="Precision Verification: Expected 4 marks"):
        prompt_to_send = "I have 72% attendance. How many attendance marks will I get?"
    if st.button("🏛️ 3. Tech Club / Cybersecurity Society", use_container_width=True, help="Process Verification: 40% batch support & Faculty Coordinator"):
        prompt_to_send = "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?"

with col2:
    if st.button("🏥 2. Ratnam Medical Leave & Timeline", use_container_width=True, help="Multi-Hop Reasoning: Yashaswini Ma'am & 7 days deadline"):
        prompt_to_send = "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?"
    if st.button("🚭 4. Smoking Fine Constraint", use_container_width=True, help="Negative Constraint: Tobacco prohibited, Disciplinary Committee, no monetary fine"):
        prompt_to_send = "How much is the fine for smoking a cigarette on campus?"

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display source evidence accordion if available
        if message.get("sources"):
            with st.expander("🔍 Official Handbook Citations & Evidence", expanded=False):
                for src in message["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <strong>Source #{src.get('source_id', 1)} — Page {src.get('page', 'N/A')}</strong><br>
                            <em>"{src.get('content', '')[:300]}..."</em>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# Chat Input Handler
user_input = st.chat_input("Ask any MirAI policy question (e.g., attendance criteria, medical leave, clubs)...")
active_query = prompt_to_send or user_input

if active_query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # Process response from backend
    with st.chat_message("assistant"):
        with st.spinner("Consulting official handbook with MultiQueryRetriever..."):
            try:
                payload = {"question": active_query, "k": 5}
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer_text = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer_text)
                    
                    if sources:
                        with st.expander("🔍 Official Handbook Citations & Evidence", expanded=False):
                            for src in sources:
                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                        <strong>Source #{src.get('source_id', 1)} — Page {src.get('page', 'N/A')}</strong><br>
                                        <em>"{src.get('content', '')[:300]}..."</em>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources
                    })
                elif response.status_code == 503:
                    error_msg = "⚠️ The policy vector database is currently empty. Please upload the MirAI Student Policy Handbook in the left sidebar first."
                    st.warning(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
                else:
                    error_msg = f"❌ Server returned error ({response.status_code}): {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
            except requests.exceptions.ConnectionError:
                error_msg = "🔌 **Backend Connection Error**: Could not reach the FastAPI server at `http://localhost:8000`. Please make sure the backend is running via `python backend.py`."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
            except requests.exceptions.Timeout:
                error_msg = "⏱️ **Server Timeout**: The backend request timed out. Please verify your Google Gemini API key quota or network connectivity."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
            except Exception as e:
                error_msg = f"⚠️ An unexpected error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
