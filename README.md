# 🔍 LegalMind: AI-Powered Legal Document Analysis

> **Transform the way you analyze legal documents with cutting-edge AI technology**

LegalMind is a sophisticated Legal RAG (Retrieval Augmented Generation) agent that revolutionizes legal document analysis by providing professional legal insights with precision and efficiency. Powered by Together AI, this intelligent system navigates complex legal terminology while maintaining the professionalism expected in legal contexts.

## ✨ Key Features

- **Expert-Level Legal Analysis** - Process and interpret complex legal documents with remarkable accuracy
- **Professional Legal Persona** - Responses crafted with the tone and expertise of a legal professional
- **Customizable Legal Context** - Tailor the agent's expertise to specific legal domains
- **Comprehensive Documentation Support** - Handles multiple document types and formats
- **Built-in Legal Disclaimer System** - Automatic inclusion of appropriate legal notices
- **Structured Legal Responses** - Get well-organized answers with proper legal context
- **Modern Web Interface** - Access powerful capabilities through an intuitive UI

## 🚀 Quick Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure Environment**
   - Create a `.env` file in the root directory
   - Add your Together AI API key:
   ```
   TOGETHER_API_KEY=your-api-key-here
   ```
   - ⚠️ Never commit your `.env` file to version control

3. **Add Your Documents**
   - Place legal documents in the `Documents` directory

## 💻 Usage Options

### Command Line Interface
```bash
python main.py
```

### Python Integration
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

## ⚙️ Powerful Configuration Options

Customize your legal assistant with `LegalRAGAgentConfig`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `document_dir` | Directory for legal documents | Required |
| `embedding_model` | Text embedding model | "togethercomputer/m2-bert-80M-8k-retrieval" |
| `generative_model` | Text generation model | "mistralai/Mixtral-8x7B-Instruct-v0.1" |
| `temperature` | Generation creativity level | 0.0 |
| `max_tokens` | Maximum response length | 512 |
| `similarity_top_k` | Documents to consider | 5 |
| `legal_persona` | Assistant's personality settings | See example |

## 📁 Project Architecture

- `rag_agent.py` - Core agent implementation with RAG capabilities
- `main.py` - Example script demonstrating agent usage
- `api.py` - FastAPI backend powering the web interface
- `static/` - Web interface assets and code
- `Documents/` - Storage for legal documents to analyze
- `.env` - Environment configuration (not in version control)

## 🔒 Security Best Practices

- Keep API keys secure and rotate regularly
- Never expose credentials in version control
- Implement proper access controls for sensitive legal documents
- Review the generated responses for sensitive information

## ⚖️ Legal Disclaimer

This AI assistant provides information based on document analysis but does not replace professional legal advice. Always consult with a qualified attorney for specific legal matters.