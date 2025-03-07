import os
from rag_agent import LegalRAGAgent, LegalRAGAgentConfig

def main():
    # Set up environment variables
    os.environ["TOGETHER_API_KEY"] = 'daacc8dc45f272f48e8571c2ff9bbccc7169541e632faa75e7efa12900cf2813'
    
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