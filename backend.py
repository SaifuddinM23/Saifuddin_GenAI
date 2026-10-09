import os
import shutil
import tempfile
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from rag_pipeline import MirAIPolicyRAG, DEFAULT_PERSIST_DIR

# Global singleton RAG instance
rag_service: Optional[MirAIPolicyRAG] = None


def get_rag_service() -> MirAIPolicyRAG:
    global rag_service
    load_dotenv(override=True)
    current_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if rag_service is None or (rag_service.api_key != current_key and current_key is not None):
        rag_service = MirAIPolicyRAG(api_key=current_key)
    return rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to automatically initialize vector store and ingest handbook if available."""
    global rag_service
    print("[MirAI Backend] Initializing Autonomous Policy Advisor Service...")
    try:
        service = get_rag_service()
        doc_count = service.get_document_count()
        print(f"[MirAI Backend] Existing vector index contains {doc_count} chunks.")
        
        # Check if index is empty and a handbook PDF exists locally
        if doc_count == 0:
            candidate_pdfs = [
                "Mirai_SoT_Policy_Handbook_2026.pdf",
                "mirai policy.pdf",
                "mirai_policy.pdf"
            ]
            for pdf_name in candidate_pdfs:
                if os.path.exists(pdf_name):
                    print(f"[MirAI Backend] Auto-ingesting default handbook: {pdf_name}")
                    result = service.ingest_pdf(pdf_name)
                    print(f"[MirAI Backend] Auto-ingestion complete: {result}")
                    break
    except Exception as e:
        print(f"[MirAI Backend] Warning during startup initialization: {e}")
    
    yield
    print("[MirAI Backend] Shutting down Policy Advisor Backend.")


# Initialize FastAPI App
app = FastAPI(
    title="Autonomous MirAI Student Policy Advisor API",
    description="Production-grade RAG backend powered by LangChain LCEL, MultiQueryRetriever, Google Gemini, and ChromaDB.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        description="The student's policy-related question.",
        example="I have 72% attendance. How many attendance marks will I get?"
    )
    k: Optional[int] = Field(
        default=5,
        description="Number of document chunks to retrieve.",
        ge=1,
        le=10
    )


class SourceItem(BaseModel):
    source_id: int
    page: int
    content: str
    metadata: Dict[str, Any] = {}


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    status: str = "success"


class IngestResponse(BaseModel):
    status: str
    message: str
    pdf_source: str
    pages_loaded: int
    chunks_created: int
    persist_directory: str


class HealthResponse(BaseModel):
    status: str
    vector_store_active: bool
    document_chunk_count: int
    llm_model: str
    embedding_model: str


# REST Endpoints
@app.get("/", tags=["General"])
async def root():
    return {
        "service": "Autonomous MirAI Student Policy Advisor",
        "status": "online",
        "docs_url": "/docs"
    }


class SetApiKeyRequest(BaseModel):
    api_key: str = Field(..., description="Google Gemini API Key starting with AIzaSy")


@app.post("/config/key", tags=["Configuration"])
async def set_api_key(req: SetApiKeyRequest):
    """Dynamically set and persist the Google Gemini API key to .env."""
    key = req.api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="API Key cannot be empty.")
    
    # Update current process env
    os.environ["GOOGLE_API_KEY"] = key
    
    # Persist to .env file
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
    
    key_found = False
    new_lines = []
    for line in lines:
        if line.startswith("GOOGLE_API_KEY="):
            new_lines.append(f"GOOGLE_API_KEY={key}\n")
            key_found = True
        else:
            new_lines.append(line)
    if not key_found:
        new_lines.insert(0, f"GOOGLE_API_KEY={key}\n")
        
    with open(env_path, "w") as f:
        f.writelines(new_lines)
        
    # Reset RAG service instance to force re-initialization
    global rag_service
    rag_service = None
    
    return {"status": "success", "message": "API key updated and persisted successfully."}


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check(service: MirAIPolicyRAG = Depends(get_rag_service)):
    """Health check endpoint to verify system status and vector store indexing."""
    count = service.get_document_count()
    return HealthResponse(
        status="healthy",
        vector_store_active=(count > 0),
        document_chunk_count=count,
        llm_model=service.llm_model_name,
        embedding_model=service.embedding_model_name
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Data Pipeline"])
async def ingest_policy_pdf(
    file: UploadFile = File(..., description="The MirAI Policy Handbook PDF file to ingest"),
    service: MirAIPolicyRAG = Depends(get_rag_service)
):
    """
    Ingests an uploaded PDF policy handbook, performs semantic chunking via
    RecursiveCharacterTextSplitter, embeds chunks with GoogleGenerativeAIEmbeddings,
    and updates the persistent ChromaDB collection.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF documents are supported."
        )

    # Save uploaded file to a temporary location
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Execute RAG ingestion pipeline
        ingest_result = service.ingest_pdf(temp_file_path)
        
        return IngestResponse(
            status="success",
            message=f"Successfully ingested and indexed {ingest_result['chunks_created']} chunks from '{file.filename}'.",
            pdf_source=file.filename,
            pages_loaded=ingest_result["pages_loaded"],
            chunks_created=ingest_result["chunks_created"],
            persist_directory=ingest_result["persist_directory"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest policy handbook: {str(e)}"
        )
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/chat", response_model=ChatResponse, tags=["RAG Reasoning"])
async def chat_with_advisor(
    request: ChatRequest,
    service: MirAIPolicyRAG = Depends(get_rag_service)
):
    """
    Accepts student queries, invokes the MultiQueryRetriever to bridge terminology gaps,
    passes retrieved context through the guarded LCEL RAG chain, and returns
    a strictly grounded answer with source citations.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    doc_count = service.get_document_count()
    if doc_count == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database is empty. Please upload/ingest the MirAI Policy Handbook first via POST /ingest."
        )

    try:
        result = service.query(request.question, k=request.k)
        
        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[
                SourceItem(
                    source_id=s["source_id"],
                    page=s["page"],
                    content=s["content"],
                    metadata=s.get("metadata", {})
                )
                for s in result.get("sources", [])
            ],
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing policy RAG query: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=True)
