import os
from dotenv import load_dotenv
from rag_agent import LegalRAGAgent, LegalRAGAgentConfig

def main():
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment variables
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("Error: TOGETHER_API_KEY environment variable is not set")
        return
    
    # Create legal agent configuration with custom persona
    config = LegalRAGAgentConfig(
        document_dir='./Documents',
        legal_persona={
            "role": "AI Legal Assistant",
            "expertise": "Legal document analysis and consultation",
            "tone": "Professional and formal",
            "context": "Specialized in analyzing legal documents and providing legal insights",
            "disclaimer": "This AI provides legal information based on available documents but does not substitute for licensed legal counsel. For specific legal advice, please consult with a qualified attorney."
        }
    )
    
    # Initialize the legal RAG agent
    agent = LegalRAGAgent(config)
    
    # Example legal queries
    queries = [
        "What is this document about and what are its legal implications?",
        "What are the key legal requirements outlined in the document?",
        "Are there any potential legal risks or compliance issues mentioned?"
    ]
    
    # Process each query
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        response = agent.query(query)
        print(f"Response:\n{response}\n")
        print("=" * 80)

if __name__ == "__main__":
    main() 