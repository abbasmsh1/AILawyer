from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List
import os
import logging
from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.llms.together import TogetherLLM

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LegalRAGAgentConfig:
    """
    Configuration for the Legal RAG Agent.
    
    Attributes:
        document_dir: Directory containing legal documents to analyze
        embedding_model: Model for text embeddings
        generative_model: Model for text generation
        temperature: Temperature for text generation (lower is more deterministic)
        max_tokens: Maximum tokens in response
        top_p: Top-p sampling parameter (1.0 = no filtering)
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens
        similarity_top_k: Number of similar documents to consider
        legal_persona: Dictionary containing the legal assistant's persona configuration
    """
    document_dir: str
    embedding_model: str = "togethercomputer/m2-bert-80M-8k-retrieval"
    generative_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    temperature: float = 0.0
    max_tokens: int = 512  # Increased for more detailed legal responses
    top_p: float = 0.9
    top_k: int = 35
    repetition_penalty: float = 1.1
    similarity_top_k: int = 5
    legal_persona: Dict[str, str] = None

    def __post_init__(self):
        """Initialize default legal persona if none is provided."""
        if self.legal_persona is None:
            self.legal_persona = {
                "role": "AI Legal Assistant",
                "expertise": "Legal analysis and consultation based on provided documentation",
                "tone": "Professional and formal",
                "context": "Specialized in analyzing legal documents and providing legal insights",
                "disclaimer": "This AI provides legal information based on available documents but does not substitute for licensed legal counsel."
            }
        
        # Validate document directory exists
        doc_path = Path(self.document_dir)
        if not doc_path.exists():
            raise ValueError(f"Document directory '{self.document_dir}' does not exist")
        if not doc_path.is_dir():
            raise ValueError(f"'{self.document_dir}' is not a directory")
        if not any(doc_path.iterdir()):
            logger.warning(f"Document directory '{self.document_dir}' is empty")


class LegalRAGAgent:
    """Agent for performing legal document analysis and consultation using RAG."""
    
    def __init__(self, config: LegalRAGAgentConfig):
        """
        Initialize the Legal RAG agent with the given configuration.
        
        Args:
            config: Configuration for the Legal RAG agent
            
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If initialization fails
        """
        self.config = config
        try:
            # Check if Together API key is set
            if not os.getenv("TOGETHER_API_KEY"):
                raise ValueError("TOGETHER_API_KEY environment variable is not set")
                
            logger.info(f"Creating service context with models: {config.embedding_model}, {config.generative_model}")
            self.service_context = self._create_service_context()
            
            logger.info(f"Loading documents from {config.document_dir}")
            self.index = self._create_index()
            logger.info("Legal RAG agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Legal RAG agent: {str(e)}")
            raise RuntimeError(f"Failed to initialize Legal RAG agent: {str(e)}") from e
    
    def _create_legal_prompt(self, query: str) -> str:
        """
        Create a legally-oriented prompt that maintains professional context.
        
        Args:
            query: The user's legal query
            
        Returns:
            str: A formatted legal prompt
        """
        legal_prompt = f"""As a legal AI assistant with expertise in document analysis, please provide a professional analysis 
        of the following query, based on the available documentation. Consider relevant legal context and implications.

        Query: {query}

        Please provide a clear, well-structured response that:
        1. Addresses the specific legal aspects of the query
        2. References relevant information from the provided documents
        3. Maintains professional legal terminology where appropriate
        4. Includes necessary context and explanations

        {self.config.legal_persona['disclaimer']}

        Analysis:"""
        return legal_prompt

    @staticmethod
    def _completion_to_prompt(completion: str) -> str:
        """
        Convert completion text to the required prompt format with legal context.
        
        Args:
            completion: Raw completion text
            
        Returns:
            str: Formatted prompt for the LLM
        """
        return f"<s>[INST] {completion} [/INST] </s>\n"
    
    def _create_service_context(self) -> ServiceContext:
        """
        Create the service context with LLM and embedding model.
        
        Returns:
            ServiceContext: Configured service context for RAG
            
        Raises:
            ValueError: If model configuration is invalid
        """
        try:
            return ServiceContext.from_defaults(
                llm=TogetherLLM(
                    model=self.config.generative_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    is_chat_model=True,
                    completion_to_prompt=self._completion_to_prompt
                ),
                embed_model=TogetherEmbedding(
                    model_name=self.config.embedding_model
                )
            )
        except Exception as e:
            logger.error(f"Failed to create service context: {str(e)}")
            raise ValueError(f"Failed to create service context: {str(e)}") from e
    
    def _create_index(self) -> VectorStoreIndex:
        """
        Create the vector store index from legal documents.
        
        Returns:
            VectorStoreIndex: Indexed documents for retrieval
            
        Raises:
            RuntimeError: If document loading or indexing fails
        """
        try:
            documents = SimpleDirectoryReader(self.config.document_dir).load_data()
            if not documents:
                logger.warning(f"No documents found in {self.config.document_dir}")
                
            return VectorStoreIndex.from_documents(
                documents,
                service_context=self.service_context
            )
        except Exception as e:
            logger.error(f"Failed to create index from documents: {str(e)}")
            raise RuntimeError(f"Failed to create index from documents: {str(e)}") from e
    
    def query(self, query_text: str) -> str:
        """
        Process a legal query using the RAG system.
        
        Args:
            query_text: The legal question or query to process
            
        Returns:
            str: The legal analysis and response from the RAG system
            
        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If query processing fails
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
            
        try:
            logger.info(f"Processing query: {query_text[:50]}...")
            legal_prompt = self._create_legal_prompt(query_text)
            query_engine = self.index.as_query_engine(
                similarity_top_k=self.config.similarity_top_k
            )
            response = query_engine.query(legal_prompt)
            
            # Format the response with the legal disclaimer
            formatted_response = f"{str(response)}\n\n{self.config.legal_persona['disclaimer']}"
            return formatted_response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise RuntimeError(f"Error processing query: {str(e)}") from e
    
    def get_legal_context(self) -> Dict[str, str]:
        """
        Get the current legal context and persona information.
        
        Returns:
            Dict[str, str]: The current legal persona configuration
        """
        return self.config.legal_persona
    
    def update_legal_context(self, new_context: Dict[str, str]) -> None:
        """
        Update the legal context and persona information.
        
        Args:
            new_context: Dictionary containing new legal context parameters
            
        Raises:
            ValueError: If the new context is invalid
        """
        if not new_context:
            raise ValueError("New context cannot be empty")
            
        logger.info("Updating legal context")
        self.config.legal_persona.update(new_context) 