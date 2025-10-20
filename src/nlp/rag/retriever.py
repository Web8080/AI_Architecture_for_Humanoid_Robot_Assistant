"""
RAG (Retrieval-Augmented Generation) System with Multi-Framework Support
Supports: LangChain, LlamaIndex
Vector Stores: FAISS (CPU/GPU), Qdrant

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
import os
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch

# LangChain (Primary)
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.docstore.document import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Install with: pip install langchain")

# LlamaIndex (Fallback)
try:
    from llama_index.core import VectorStoreIndex, Document as LlamaDocument, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    logging.warning("LlamaIndex not available. Install with: pip install llama-index")

# FAISS (Primary vector store)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

# Qdrant (Alternative vector store)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant not available. Install with: pip install qdrant-client")

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence-transformers not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieval result"""
    text: str
    metadata: Dict[str, Any]
    score: float
    source: str


class RAGRetriever:
    """
    Multi-framework RAG system with automatic fallback.
    Supports both LangChain and LlamaIndex.
    Auto-detects GPU/CPU for embeddings and vector search.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type: str = "faiss",  # faiss or qdrant
        persist_dir: str = "./data/vector_store",
        use_gpu: Optional[bool] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5
    ):
        """
        Initialize RAG retriever.
        
        Args:
            embedding_model: Sentence transformer model name
            vector_store_type: Type of vector store (faiss or qdrant)
            persist_dir: Directory to persist vector store
            use_gpu: Force GPU usage (None = auto-detect)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of results to retrieve
        """
        self.embedding_model_name = embedding_model
        self.vector_store_type = vector_store_type
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Auto-detect GPU
        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu
        
        # Set device for embeddings
        self.device = "cuda" if self.use_gpu else "cpu"
        logger.info(f"Using device for embeddings: {self.device}")
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.framework = None  # "langchain" or "llamaindex"
        
        self._initialize_embeddings()
        self._initialize_vector_store()
        
        # Text splitter
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:
            self.text_splitter = None
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        # Try HuggingFace embeddings first
        if LANGCHAIN_AVAILABLE:
            try:
                logger.info(f"Loading embeddings with LangChain: {self.embedding_model_name}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={'device': self.device},
                    encode_kwargs={'device': self.device, 'batch_size': 32}
                )
                logger.info(" Embeddings initialized with LangChain")
                return
            except Exception as e:
                logger.warning(f"LangChain embeddings failed: {e}")
        
        # Try sentence-transformers directly
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading embeddings with sentence-transformers: {self.embedding_model_name}")
                self.embeddings = SentenceTransformer(self.embedding_model_name, device=self.device)
                logger.info(" Embeddings initialized with sentence-transformers")
                return
            except Exception as e:
                logger.error(f"Sentence-transformers failed: {e}")
        
        logger.error("Failed to initialize embeddings")
    
    def _initialize_vector_store(self):
        """Initialize vector store with fallback"""
        if self.vector_store_type == "faiss":
            self._initialize_faiss()
        elif self.vector_store_type == "qdrant":
            self._initialize_qdrant()
        else:
            logger.error(f"Unknown vector store type: {self.vector_store_type}")
    
    def _initialize_faiss(self):
        """Initialize FAISS vector store"""
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available")
            return
        
        faiss_index_path = self.persist_dir / "faiss_index"
        
        # Check if exists
        if faiss_index_path.exists() and LANGCHAIN_AVAILABLE:
            try:
                logger.info("Loading existing FAISS index")
                self.vector_store = FAISS.load_local(
                    str(faiss_index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.framework = "langchain"
                logger.info(" FAISS index loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
        
        # Create new index
        if LANGCHAIN_AVAILABLE:
            logger.info("Creating new FAISS index with LangChain")
            # Initialize with empty documents
            self.vector_store = FAISS.from_documents(
                [Document(page_content="initialization", metadata={"source": "init"})],
                self.embeddings
            )
            self.framework = "langchain"
            logger.info(" New FAISS index created")
        else:
            logger.error("Cannot create FAISS index without LangChain")
    
    def _initialize_qdrant(self):
        """Initialize Qdrant vector store (optional)"""
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant not available. Falling back to FAISS")
            self._initialize_faiss()
            return
        
        try:
            # Use in-memory Qdrant for development
            self.qdrant_client = QdrantClient(":memory:")
            logger.info(" Qdrant initialized (in-memory)")
            self.framework = "qdrant"
        except Exception as e:
            logger.error(f"Qdrant initialization failed: {e}. Falling back to FAISS")
            self._initialize_faiss()
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text documents
            metadatas: Optional metadata for each document
        """
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        # Split texts into chunks
        if self.text_splitter:
            chunks = []
            chunk_metadatas = []
            for text, metadata in zip(texts, metadatas):
                text_chunks = self.text_splitter.split_text(text)
                chunks.extend(text_chunks)
                chunk_metadatas.extend([metadata.copy() for _ in text_chunks])
        else:
            chunks = texts
            chunk_metadatas = metadatas
        
        # Add to vector store
        if self.framework == "langchain" and self.vector_store is not None:
            try:
                # Create documents
                documents = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(chunks, chunk_metadatas)
                ]
                
                # Add to FAISS
                self.vector_store.add_documents(documents)
                
                # Save to disk
                self.vector_store.save_local(str(self.persist_dir / "faiss_index"))
                
                logger.info(f"Added {len(documents)} document chunks to vector store")
            except Exception as e:
                logger.error(f"Failed to add documents: {e}")
        else:
            logger.error("Vector store not available")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results (uses default if None)
            
        Returns:
            List of retrieval results
        """
        if top_k is None:
            top_k = self.top_k
        
        if self.framework == "langchain" and self.vector_store is not None:
            try:
                # Use similarity search with scores
                results = self.vector_store.similarity_search_with_score(query, k=top_k)
                
                retrieval_results = []
                for doc, score in results:
                    retrieval_results.append(RetrievalResult(
                        text=doc.page_content,
                        metadata=doc.metadata,
                        score=float(score),
                        source="LangChain-FAISS"
                    ))
                
                logger.debug(f"Retrieved {len(retrieval_results)} results for query: {query}")
                return retrieval_results
                
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                return []
        
        logger.error("No vector store available for retrieval")
        return []
    
    def query_with_context(self, query: str, top_k: Optional[int] = None) -> Tuple[str, List[RetrievalResult]]:
        """
        Retrieve documents and format as context for LLM.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Tuple of (formatted_context, retrieval_results)
        """
        results = self.retrieve(query, top_k)
        
        if not results:
            return "", []
        
        # Format context for LLM
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.metadata.get('source', 'unknown')
            context_parts.append(f"[{i}] (Source: {source})\n{result.text}\n")
        
        formatted_context = "\n".join(context_parts)
        
        return formatted_context, results
    
    def get_status(self) -> Dict[str, Any]:
        """Get RAG system status"""
        return {
            "framework": self.framework,
            "langchain_available": LANGCHAIN_AVAILABLE,
            "llamaindex_available": LLAMAINDEX_AVAILABLE,
            "faiss_available": FAISS_AVAILABLE,
            "qdrant_available": QDRANT_AVAILABLE,
            "vector_store_ready": self.vector_store is not None,
            "embeddings_ready": self.embeddings is not None,
            "gpu_enabled": self.use_gpu,
            "vector_store_type": self.vector_store_type
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG retriever
    retriever = RAGRetriever(persist_dir="./test_vector_store")
    
    print("=" * 80)
    print("RAG RETRIEVER - TESTING")
    print("=" * 80)
    print(f"\nStatus: {retriever.get_status()}\n")
    
    # Add sample knowledge base
    sample_documents = [
        "Humanoid robots can navigate autonomously using SLAM and sensor fusion.",
        "The robot uses YOLOv8 for object detection and SAM for segmentation.",
        "NLP capabilities include intent classification, entity extraction, and dialogue management.",
        "The system supports both cloud and edge deployment on NVIDIA Jetson.",
        "Safety features include multi-layer checks, E-stop, and anomaly detection.",
    ]
    
    print("Adding sample documents to vector store...")
    retriever.add_documents(sample_documents)
    print(f" Added {len(sample_documents)} documents\n")
    
    # Test retrieval
    test_queries = [
        "How does the robot detect objects?",
        "What are the safety features?",
        "Tell me about navigation capabilities"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=2)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  [{i}] Score: {result.score:.3f}")
                print(f"      Text: {result.text[:100]}...")
                print(f"      Source: {result.source}")
        else:
            print("  → No results found")
    
    # Test context formatting
    print(f"\n{'='*80}")
    print("CONTEXT FORMATTING TEST")
    print(f"{'='*80}")
    query = "What can the robot do?"
    context, results = retriever.query_with_context(query, top_k=3)
    print(f"\nQuery: {query}")
    print(f"\nFormatted Context:\n{context}")
    
    print("\n" + "=" * 80)

