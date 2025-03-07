# Legal Document Analysis RAG Agent

This project implements a specialized Legal RAG (Retrieval Augmented Generation) agent that can analyze legal documents and provide professional legal insights based on the available documentation. The agent maintains a professional legal persona while analyzing documents using the Together AI platform.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables:
   - Create a `.env` file in the root directory
   - Add your Together AI API key:
   ```
   TOGETHER_API_KEY=your-api-key-here
   ```
   - Never commit your `.env` file to version control

3. Place your legal documents in the `Documents` directory.

## Usage

You can use the Legal RAG agent in two ways:

1. Using the main script:
```bash
python main.py
```

2. Using the Legal RAG agent in your own code:
```python
from rag_agent import LegalRAGAgent, LegalRAGAgentConfig

# Create configuration with custom legal persona
config = LegalRAGAgentConfig(
    document_dir='./Documents',
    legal_persona={
        "role": "AI Legal Assistant",
        "expertise": "Legal document analysis",
        "tone": "Professional and formal",
        "context": "Specialized in legal document analysis",
        "disclaimer": "This AI provides legal information but does not substitute for licensed legal counsel."
    }
)

# Initialize agent
agent = LegalRAGAgent(config)

# Make legal queries
response = agent.query("What are the legal implications of this document?")
print(response)

# Update legal context if needed
agent.update_legal_context({
    "expertise": "Contract law and compliance",
    "context": "Focused on contractual obligations and compliance requirements"
})
```

## Configuration

The `LegalRAGAgentConfig` class allows you to customize various parameters:

- `document_dir`: Directory containing the legal documents to analyze
- `embedding_model`: Model for text embeddings (default: "togethercomputer/m2-bert-80M-8k-retrieval")
- `generative_model`: Model for text generation (default: "mistralai/Mixtral-8x7B-Instruct-v0.1")
- `temperature`: Temperature for text generation (default: 0.0)
- `max_tokens`: Maximum tokens in response (default: 512)
- `top_p`, `top_k`: Sampling parameters
- `similarity_top_k`: Number of similar documents to consider (default: 5)
- `legal_persona`: Dictionary containing the legal assistant's persona configuration:
  - `role`: The role of the AI assistant
  - `expertise`: Areas of legal expertise
  - `tone`: Communication style
  - `context`: Specialization context
  - `disclaimer`: Legal disclaimer message

## Features

- Professional legal document analysis
- Maintains legal context and terminology
- Customizable legal persona and expertise
- Automatic inclusion of legal disclaimers
- Structured legal responses with proper context
- Support for multiple document types

## Project Structure

- `rag_agent.py`: Contains the main Legal RAG agent implementation
- `main.py`: Example script showing how to use the legal agent
- `api.py`: FastAPI backend for the web interface
- `static/`: Directory containing the web interface files
- `requirements.txt`: Project dependencies
- `Documents/`: Directory for storing legal documents to analyze
- `.env`: Environment variables (not in version control)

## Environment Variables

The following environment variables are required:

- `TOGETHER_API_KEY`: Your Together AI API key

You can set these variables in a `.env` file or in your system's environment.

## Security Notes

- Never commit your `.env` file or API keys to version control
- Keep your API keys secure and rotate them regularly
- Use appropriate access controls for your legal documents

## Legal Disclaimer

This AI assistant provides legal information based on available documents but does not substitute for licensed legal counsel. For specific legal advice, please consult with a qualified attorney.