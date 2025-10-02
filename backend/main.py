"""
FastAPI Backend for Interview Transcript Analyzer
Main application file that handles API endpoints for analyzing interview transcripts
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import sqlite3
from datetime import datetime
import json

# Initialize FastAPI app
app = FastAPI(
    title="Interview Transcript Analyzer",
    description="AI-powered interview transcript analysis and insights",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = "interviews.db"

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Pydantic models
class TranscriptAnalysis(BaseModel):
    transcript_id: str
    transcript_text: str
    analysis_results: dict
    created_at: datetime

class AnalysisRequest(BaseModel):
    transcript_text: str
    analysis_type: str = "comprehensive"

class AnalysisResponse(BaseModel):
    transcript_id: str
    analysis_results: dict
    status: str
    message: str

# Database initialization
def init_database():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcripts (
            id TEXT PRIMARY KEY,
            transcript_text TEXT NOT NULL,
            analysis_results TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "Interview Transcript Analyzer API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_transcript(
    request: AnalysisRequest,
    db: sqlite3.Connection = Depends(get_db)
):
    """
    Analyze interview transcript and return insights
    """
    try:
        # Generate unique ID for this analysis
        transcript_id = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Placeholder analysis results (replace with actual AI analysis)
        analysis_results = {
            "sentiment_analysis": {
                "overall_sentiment": "positive",
                "confidence": 0.85,
                "key_emotions": ["confidence", "enthusiasm", "professionalism"]
            },
            "key_topics": [
                "technical skills",
                "problem solving",
                "team collaboration",
                "leadership experience"
            ],
            "strengths": [
                "Clear communication",
                "Technical expertise",
                "Problem-solving approach"
            ],
            "areas_for_improvement": [
                "Could provide more specific examples",
                "Consider elaborating on leadership experience"
            ],
            "overall_score": 8.5,
            "recommendations": [
                "Strong candidate with good technical background",
                "Consider for next round of interviews"
            ]
        }
        
        # Store in database
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO transcripts (id, transcript_text, analysis_results) VALUES (?, ?, ?)",
            (transcript_id, request.transcript_text, json.dumps(analysis_results))
        )
        db.commit()
        
        return AnalysisResponse(
            transcript_id=transcript_id,
            analysis_results=analysis_results,
            status="success",
            message="Transcript analyzed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/transcripts")
async def get_transcripts(db: sqlite3.Connection = Depends(get_db)):
    """Get all stored transcripts"""
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM transcripts ORDER BY created_at DESC")
        transcripts = cursor.fetchall()
        
        return {
            "transcripts": [
                {
                    "id": row["id"],
                    "transcript_text": row["transcript_text"][:200] + "..." if len(row["transcript_text"]) > 200 else row["transcript_text"],
                    "analysis_results": json.loads(row["analysis_results"]) if row["analysis_results"] else None,
                    "created_at": row["created_at"]
                }
                for row in transcripts
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve transcripts: {str(e)}")

@app.get("/api/transcripts/{transcript_id}")
async def get_transcript(
    transcript_id: str,
    db: sqlite3.Connection = Depends(get_db)
):
    """Get specific transcript by ID"""
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM transcripts WHERE id = ?", (transcript_id,))
        transcript = cursor.fetchone()
        
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found")
        
        return {
            "id": transcript["id"],
            "transcript_text": transcript["transcript_text"],
            "analysis_results": json.loads(transcript["analysis_results"]) if transcript["analysis_results"] else None,
            "created_at": transcript["created_at"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve transcript: {str(e)}")

@app.delete("/api/transcripts/{transcript_id}")
async def delete_transcript(
    transcript_id: str,
    db: sqlite3.Connection = Depends(get_db)
):
    """Delete a transcript"""
    try:
        cursor = db.cursor()
        cursor.execute("DELETE FROM transcripts WHERE id = ?", (transcript_id,))
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Transcript not found")
        
        db.commit()
        return {"message": "Transcript deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete transcript: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
