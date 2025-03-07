from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.llms.together import TogetherLLM


@dataclass
class LegalRAGAgentConfig:
    """Configuration for the Legal RAG Agent."""
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
        if self.legal_persona is None:
            self.legal_persona = {
                "role": "AI Legal Assistant",
                "expertise": "Legal analysis and consultation based on provided documentation",
                "tone": "Professional and formal",
                "context": "Specialized in analyzing legal documents and providing legal insights",
                "disclaimer": "This AI provides legal information based on available documents but does not substitute for licensed legal counsel."
            }


class LegalRAGAgent:
    """Agent for performing legal document analysis and consultation using RAG."""
    
    def __init__(self, config: LegalRAGAgentConfig):
        """Initialize the Legal RAG agent with the given configuration."""
        self.config = config
        self.service_context = self._create_service_context()
        self.index = self._create_index()
    
    def _create_legal_prompt(self, query: str) -> str:
        """Create a legally-oriented prompt that maintains professional context."""
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
        """Convert completion text to the required prompt format with legal context."""
        return f"<s>[INST] {completion} [/INST] </s>\n"
    
    def _create_service_context(self) -> ServiceContext:
        """Create the service context with LLM and embedding model."""
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
    
    def _create_index(self) -> VectorStoreIndex:
        """Create the vector store index from legal documents."""
        documents = SimpleDirectoryReader(self.config.document_dir).load_data()
        return VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context
        )
    
    def query(self, query_text: str) -> str:
        """
        Process a legal query using the RAG system.
        
        Args:
            query_text: The legal question or query to process
            
        Returns:
            str: The legal analysis and response from the RAG system
        """
        legal_prompt = self._create_legal_prompt(query_text)
        query_engine = self.index.as_query_engine(
            similarity_top_k=self.config.similarity_top_k
        )
        response = query_engine.query(legal_prompt)
        
        # Format the response with the legal disclaimer
        formatted_response = f"{str(response)}\n\n{self.config.legal_persona['disclaimer']}"
        return formatted_response
    
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
        """
        self.config.legal_persona.update(new_context) 