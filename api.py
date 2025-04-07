import os
import shutil
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

from rag_agent import LegalRAGAgent, LegalRAGAgentConfig

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Legal Document Analysis API",
    description="API for analyzing legal documents using RAG",
    version="1.0.0"
)

# Add CORS middleware with more restrictive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "https://localhost:8000"],  # Restrict to specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT"],  # Only allow necessary methods
    allow_headers=["*"],
)

# Get API key from environment variables
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    logger.error("TOGETHER_API_KEY environment variable is not set")
    raise ValueError("TOGETHER_API_KEY environment variable is not set")

# Initialize the Legal RAG agent
try:
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
    logger.info("Legal RAG agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Legal RAG agent: {str(e)}")
    raise

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
    try:
        with open("static/index.html") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("index.html not found in static directory")
        raise HTTPException(status_code=404, detail="Frontend not found")

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a legal query and return the response."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        logger.info(f"Processing query: {request.query[:50]}...")
        response = agent.query(request.query)
        legal_context = agent.get_legal_context()
        return QueryResponse(response=response, legal_context=legal_context)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.put("/api/context")
async def update_context(request: UpdateContextRequest):
    """Update the legal context of the agent."""
    if not request.legal_context:
        raise HTTPException(status_code=400, detail="Legal context cannot be empty")
        
    try:
        logger.info("Updating legal context")
        agent.update_legal_context(request.legal_context)
        return {"status": "success", "message": "Legal context updated successfully"}
    except Exception as e:
        logger.error(f"Error updating context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating context: {str(e)}")

@app.get("/api/context")
async def get_context():
    """Get the current legal context."""
    try:
        return {"legal_context": agent.get_legal_context()}
    except Exception as e:
        logger.error(f"Error getting context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting context: {str(e)}")

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the Documents directory.
    
    Supported file types: PDF, DOCX, TXT
    """
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.txt', '.doc', '.rtf']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Create the documents directory if it doesn't exist
        os.makedirs("./Documents", exist_ok=True)
        
        # Save the file
        file_path = os.path.join("./Documents", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Reinitialize the agent to include the new document
        global agent
        agent = LegalRAGAgent(config)
        logger.info(f"Document uploaded: {file.filename}")
        
        return JSONResponse(
            content={
                "status": "success", 
                "message": f"Document '{file.filename}' uploaded successfully",
                "filename": file.filename
            },
            status_code=200
        )
    except HTTPException as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    """List all documents in the Documents directory."""
    try:
        documents_dir = "./Documents"
        os.makedirs(documents_dir, exist_ok=True)
        
        files = []
        for filename in os.listdir(documents_dir):
            file_path = os.path.join(documents_dir, filename)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                files.append({
                    "name": filename,
                    "size": file_size,
                    "size_formatted": format_file_size(file_size)
                })
        
        return {"documents": files}
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document from the Documents directory."""
    try:
        # Normalize and validate filename
        filename = os.path.basename(filename)  # Prevent path traversal
        file_path = os.path.join("./Documents", filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
        
        # Delete file
        os.remove(file_path)
        
        # Reinitialize the agent after file deletion
        global agent
        agent = LegalRAGAgent(config)
        logger.info(f"Document deleted: {filename}")
        
        return JSONResponse(
            content={
                "status": "success", 
                "message": f"Document '{filename}' deleted successfully"
            },
            status_code=200
        )
    except HTTPException as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

def format_file_size(size_in_bytes):
    """Format file size in bytes to a readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 