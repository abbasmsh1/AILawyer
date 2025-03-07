import os
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_agent import LegalRAGAgent, LegalRAGAgentConfig

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Legal Document Analysis API",
    description="API for analyzing legal documents using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment variables
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("TOGETHER_API_KEY environment variable is not set")

# Set API key in environment
os.environ["TOGETHER_API_KEY"] = api_key

# Initialize the Legal RAG agent
config = LegalRAGAgentConfig(
    document_dir='./Documents',
    legal_persona={
        "role": "AI Legal Assistant",
        "expertise": "Legal document analysis and consultation",
        "tone": "Professional and formal",
        "context": "Specialized in analyzing legal documents and providing legal insights",
        "disclaimer": "This AI provides legal information based on available documents but does not substitute for licensed legal counsel."
    }
)

agent = LegalRAGAgent(config)

# Define request/response models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    legal_context: Dict[str, str]

class UpdateContextRequest(BaseModel):
    legal_context: Dict[str, str]

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML interface."""
    with open("static/index.html") as f:
        return f.read()

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a legal query and return the response."""
    try:
        response = agent.query(request.query)
        legal_context = agent.get_legal_context()
        return QueryResponse(response=response, legal_context=legal_context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/context")
async def update_context(request: UpdateContextRequest):
    """Update the legal context of the agent."""
    try:
        agent.update_legal_context(request.legal_context)
        return {"status": "success", "message": "Legal context updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/context")
async def get_context():
    """Get the current legal context."""
    try:
        return {"legal_context": agent.get_legal_context()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 