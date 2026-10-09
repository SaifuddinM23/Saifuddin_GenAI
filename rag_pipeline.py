import os
import shutil
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Configuration constants
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
DEFAULT_LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# System Prompt Guardrail: Strictly factual, zero hallucination
SYSTEM_PROMPT_TEMPLATE = """You are the Autonomous MirAI Student Policy Advisor, an authoritative AI assistant representing the MirAI School of Technology (MSOT) administration.

CRITICAL INSTRUCTIONS & GUARDRAILS:
1. Strict Factual Grounding: Answer the student's question using ONLY the factual context provided below from the official MirAI Student Policy Handbook.
2. Zero Hallucination: NEVER rely on assumptions, external general knowledge, or unstated institutional policies. Do NOT fabricate dates, people, monetary fines, or rules under any circumstances.
3. Negative Constraint Handling: If the provided context does not contain sufficient facts to answer the question, or if asked about something not defined in the policy (such as a specific fine amount that does not exist in the policy), explicitly state what the policy says and politely decline to invent details (e.g. "I cannot find any specific monetary fine mentioned in the official MirAI Policy Handbook. However, the policy strictly states that...").
4. Precision & Accuracy: Include exact numbers, thresholds (e.g. attendance tiers, percentages, submission day limits), specific campus roles/names (e.g. Campus Managers, Faculty Coordinators), and formal procedures whenever present in the context.
5. Tone: Maintain a professional, supportive, clear, and administrative tone.

CONTEXT FROM MIRAI POLICY HANDBOOK:
----------------------------------------
{context}
----------------------------------------

STUDENT QUESTION:
{question}

OFFICIAL POLICY ADVISOR ANSWER:"""

# Custom MultiQuery generation prompt to bridge informal student queries and formal handbook vocabulary
MULTI_QUERY_PROMPT_TEMPLATE = """You are an AI language model assistant assisting university students with the MirAI School of Technology Policy Handbook.
Your task is to generate 3 different versions of the given student query to retrieve relevant policy documents from a vector database.
By generating multiple perspectives on the user query, your goal is to help overcome vocabulary gaps between casual student terminology (e.g., "bunking", "fine", "sick", "starting club") and formal institutional policy terminology (e.g., "attendance debarment", "disciplinary action", "medical leave 7-day rule", "student organization proposal threshold").

Provide these alternative queries separated by newlines. Do not include numbering or prefixes.

Original Student Query: {question}

Alternative Queries:"""


class MirAIPolicyRAG:
    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        llm_model_name: str = DEFAULT_LLM_MODEL,
        api_key: Optional[str] = None
    ):
        # Reload .env dynamically in case user updated it recently
        load_dotenv(override=True)
        
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model_name,
            google_api_key=self.api_key
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.llm_model_name,
            temperature=0.0,
            google_api_key=self.api_key
        )
        
        self.vector_store: Optional[Chroma] = None
        self._init_vector_store()

    def _init_vector_store(self):
        """Initialize connection to existing Chroma vector store if present."""
        if os.path.exists(self.persist_directory):
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="mirai_policy_handbook"
                )
            except Exception as e:
                print(f"Warning initializing existing ChromaDB: {e}")
                self.vector_store = None

    def ingest_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Loads a PDF via PyPDFLoader, splits text with RecursiveCharacterTextSplitter,
        and indexes into ChromaDB.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
        # 1. Document Loading
        loader = PyPDFLoader(file_path=pdf_path)
        raw_documents = loader.load()
        
        # 2. Semantic Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "•", "●", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(raw_documents)
        
        # Add metadata enhancements (ensure page numbers are 1-indexed for display)
        for chunk in chunks:
            chunk.metadata["source_file"] = os.path.basename(pdf_path)
            if "page" in chunk.metadata:
                chunk.metadata["page_number"] = chunk.metadata["page"] + 1
            else:
                chunk.metadata["page_number"] = 1

        # 3. Vector Database Storage in ChromaDB
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="mirai_policy_handbook"
        )
        
        return {
            "status": "success",
            "pdf_source": os.path.basename(pdf_path),
            "pages_loaded": len(raw_documents),
            "chunks_created": len(chunks),
            "persist_directory": self.persist_directory
        }

    def get_document_count(self) -> int:
        """Returns total number of chunks indexed in ChromaDB."""
        if self.vector_store is None:
            self._init_vector_store()
        if self.vector_store is None:
            return 0
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    def get_retriever(self, k: int = 5):
        """Builds and returns a MultiQueryRetriever on top of ChromaDB."""
        if self.vector_store is None:
            self._init_vector_store()
        if self.vector_store is None:
            raise ValueError("Vector store is empty or not initialized. Please ingest a handbook PDF first.")

        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        multi_query_prompt = PromptTemplate(
            input_variables=["question"],
            template=MULTI_QUERY_PROMPT_TEMPLATE
        )

        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
            prompt=multi_query_prompt
        )
        
        return multi_query_retriever

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Executes the advanced LCEL RAG chain using MultiQueryRetriever and returns
        the strictly grounded answer along with source context chunks and metadata.
        """
        retriever = self.get_retriever(k=k)
        
        # Retrieve documents using MultiQueryRetriever
        retrieved_docs: List[Document] = retriever.invoke(question)
        
        if not retrieved_docs:
            return {
                "question": question,
                "answer": "I cannot find any relevant policy in the official MirAI Student Policy Handbook regarding your query. Please reach out to the Student Helpdesk at studenthelpdesk@msot.org for formal guidance.",
                "sources": [],
                "context_used": ""
            }

        # Format context for prompt
        formatted_context_list = []
        sources = []
        for i, doc in enumerate(retrieved_docs):
            page = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
            source_file = doc.metadata.get("source_file", "MirAI Policy Handbook")
            formatted_context_list.append(
                f"[Source #{i+1} | Page {page} | Document: {source_file}]\n{doc.page_content.strip()}"
            )
            sources.append({
                "source_id": i + 1,
                "page": page,
                "content": doc.page_content.strip(),
                "metadata": doc.metadata
            })
            
        full_context_text = "\n\n".join(formatted_context_list)
        
        # LCEL Prompt & Generation
        prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
        
        chain = prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": full_context_text,
            "question": question
        })
        
        return {
            "question": question,
            "answer": answer.strip(),
            "sources": sources,
            "context_used": full_context_text
        }
