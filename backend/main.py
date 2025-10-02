"""
FastAPI Backend for Interview Transcript Analysis
Provides endpoints for analyzing interview transcripts using ChatGPT integration
"""

from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import requests
import json
import time
import re
from datetime import datetime
import logging
import uuid
import sqlite3
from contextlib import contextmanager
from dotenv import load_dotenv
import openai
import PyPDF2
import io

def clean_extracted_text(text: str) -> str:
    """
    Clean and format text extracted from PDF
    
    Args:
        text: Raw text from PDF extraction
    
    Returns:
        Cleaned and formatted text
    """
    if not text:
        return ""
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF extraction issues - join words that were split
    text = re.sub(r'(\w)\s+(\w)', r'\1 \2', text)
    
    # Add proper line breaks for speaker changes (e.g., "James:", "Interviewer:")
    text = re.sub(r'([A-Z][a-z]+:)\s*', r'\n\1 ', text)
    
    # Add line breaks for timestamps (e.g., "[00:20]", "[00:01:30]")
    text = re.sub(r'(\[\d{1,2}:\d{2}(?::\d{2})?\])', r'\n\1', text)
    
    # Fix common word splitting issues
    text = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', text)  # Fix hyphenated words
    text = re.sub(r'(\w)\s*\.\s*(\w)', r'\1. \2', text)  # Fix sentence endings
    
    # Fix common interview transcript issues
    text = re.sub(r'(\w)\s*,\s*(\w)', r'\1, \2', text)  # Fix comma spacing
    text = re.sub(r'(\w)\s*:\s*(\w)', r'\1: \2', text)  # Fix colon spacing
    
    # Join single words that should be part of sentences
    lines = text.split('\n')
    cleaned_lines = []
    current_line = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_line:
                cleaned_lines.append(current_line)
                current_line = ""
            continue
            
        # If line is just a single word and doesn't end with punctuation, it might be part of a sentence
        if len(line.split()) == 1 and not line.endswith(('.', '!', '?', ':', ']')):
            current_line += " " + line if current_line else line
        else:
            if current_line:
                current_line += " " + line
                cleaned_lines.append(current_line)
                current_line = ""
            else:
                cleaned_lines.append(line)
    
    if current_line:
        cleaned_lines.append(current_line)
    
    text = '\n'.join(cleaned_lines)
    
    # Clean up multiple line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Ensure proper spacing around punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'([.,!?])\s*([A-Z])', r'\1 \2', text)
    
    return text

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Interview Transcript Analyzer",
    description="AI-powered interview transcript analysis using ChatGPT",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_PATH = "interviews.db"

def init_database():
    """Initialize the SQLite database with required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interviews (
                id TEXT PRIMARY KEY,
                title TEXT UNIQUE NOT NULL,
                transcript TEXT NOT NULL,
                summary TEXT,
                highlights TEXT,
                lowlights TEXT,
                key_entities TEXT,
                timeline TEXT,
                analysis_metadata TEXT,
                upload_method TEXT DEFAULT 'manual',
                original_filename TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add upload_method column if it doesn't exist (for existing databases)
        try:
            cursor.execute('ALTER TABLE interviews ADD COLUMN upload_method TEXT DEFAULT "manual"')
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Add original_filename column if it doesn't exist
        try:
            cursor.execute('ALTER TABLE interviews ADD COLUMN original_filename TEXT')
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.commit()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Initialize database on startup
init_database()

# Pydantic models for request/response
class TranscriptRequest(BaseModel):
    """Request model for transcript analysis"""
    title: str = Field(..., description="Unique title for the interview")
    transcript: str = Field(..., description="The interview transcript text to analyze")

class TimelineItem(BaseModel):
    """Model for timeline items"""
    timestamp: str = Field(..., description="Timestamp in HH:MM:SS format")
    topic: str = Field(..., description="Topic or section name")
    description: str = Field(..., description="Brief description of the section")

class AnalysisResponse(BaseModel):
    """Response model for transcript analysis"""
    interview_id: str = Field(..., description="Unique interview ID")
    title: str = Field(..., description="Interview title")
    summary: str = Field(..., description="Overall summary of the interview")
    highlights: List[str] = Field(..., description="Key highlights from the interview")
    lowlights: List[str] = Field(..., description="Areas of concern or lowlights")
    key_entities: List[str] = Field(..., description="Key names, companies, technologies mentioned")
    timeline: List[TimelineItem] = Field(..., description="Chronological breakdown of the interview")
    analysis_metadata: Dict[str, Any] = Field(..., description="Metadata about the analysis")
    upload_method: str = Field(..., description="Upload method (manual or file)")
    original_filename: Optional[str] = Field(None, description="Original filename if uploaded via file")
    created_at: str = Field(..., description="Creation timestamp")

class InterviewListItem(BaseModel):
    """Model for interview list items"""
    interview_id: str = Field(..., description="Unique interview ID")
    title: str = Field(..., description="Interview title")
    created_at: str = Field(..., description="Creation timestamp")
    summary: str = Field(..., description="Brief summary")
    upload_method: str = Field(..., description="Upload method (manual or file)")
    original_filename: Optional[str] = Field(None, description="Original filename if uploaded via file")

class SearchResponse(BaseModel):
    """Response model for search results"""
    interviews: List[InterviewListItem] = Field(..., description="List of matching interviews")
    total_count: int = Field(..., description="Total number of matches")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")

# ChatGPT API configuration
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-3.5-turbo"

# Get API key from environment only
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure OpenAI client
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI API key loaded successfully")
else:
    logger.warning("No OpenAI API key available. ChatGPT features will use fallback analysis.")

# Rate limiting
last_request_time = 0
MIN_REQUEST_INTERVAL = 3.0  # Increased to 3 seconds to respect OpenAI rate limits

def call_chatgpt_api(api_key: str, messages: List[Dict[str, str]], model: str = DEFAULT_MODEL) -> str:
    """
    Make a call to ChatGPT API using OpenAI library
    
    Args:
        api_key: OpenAI API key (not used directly, configured globally)
        messages: List of message dictionaries
        model: GPT model to use
    
    Returns:
        Response content from ChatGPT
    """
    global last_request_time
    
    # Rate limiting - ensure minimum interval between requests
    current_time = time.time()
    time_since_last = current_time - last_request_time
    if time_since_last < MIN_REQUEST_INTERVAL:
        sleep_time = MIN_REQUEST_INTERVAL - time_since_last
        logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
    
    last_request_time = time.time()
    
    max_retries = 2
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Making ChatGPT API request (attempt {attempt + 1}/{max_retries})")
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )
            
            logger.info("ChatGPT API request successful!")
            return response.choices[0].message["content"]
                
        except openai.error.RateLimitError as e:
            if attempt == max_retries - 1:
                logger.error(f"Rate limit exceeded after {max_retries} attempts")
                raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded. Please wait a moment and try again.")
            else:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                
        except openai.error.AuthenticationError as e:
            logger.error("Authentication failed - check API key")
            raise HTTPException(status_code=401, detail="OpenAI API authentication failed. Please check your API key.")
            
        except openai.error.APIError as e:
            if attempt == max_retries - 1:
                logger.error(f"OpenAI API error after {max_retries} attempts: {str(e)}")
                raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
            else:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
                time.sleep(retry_delay)
                
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Unexpected error after {max_retries} attempts: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
            else:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
                time.sleep(retry_delay)
    
    # If we get here, all retries failed
    logger.error("ChatGPT API failed after all retries, using fallback analysis")
    raise Exception("ChatGPT API unavailable - using fallback analysis")

def analyze_transcript_comprehensive(api_key: str, transcript: str) -> dict:
    """
    Analyze transcript comprehensively in a single API call
    
    Args:
        api_key: OpenAI API key
        transcript: Interview transcript text
    
    Returns:
        Dictionary with summary, highlights, lowlights, entities, and timeline
    """
    prompt = f"""
    Analyze the following interview transcript comprehensively. Provide your response in this exact JSON format:

    {{
        "summary": "Provide a detailed, comprehensive summary of the interview in 4-6 sentences. Include the candidate's background, key experiences, specific projects mentioned, technical skills, leadership approach, challenges faced, and overall assessment. Make it thorough and professional like a detailed interview report.",
        "highlights": ["highlight1", "highlight2", "highlight3", "highlight4", "highlight5"],
        "lowlights": ["lowlight1", "lowlight2", "lowlight3"],
        "key_entities": ["entity1", "entity2", "entity3", "entity4", "entity5", "entity6", "entity7"],
        "timeline": [
            {{
                "timestamp": "00:00:00",
                "topic": "Introduction",
                "description": "Brief description"
            }},
            {{
                "timestamp": "00:02:30",
                "topic": "Background Discussion",
                "description": "Brief description"
            }},
            {{
                "timestamp": "00:05:00",
                "topic": "Technical Discussion",
                "description": "Brief description"
            }},
            {{
                "timestamp": "00:07:30",
                "topic": "Project Discussion",
                "description": "Brief description"
            }},
            {{
                "timestamp": "00:10:00",
                "topic": "Conclusion",
                "description": "Brief description"
            }}
        ]
    }}

    Focus on:
    - Summary: Detailed analysis including background, experience, projects, skills, leadership, challenges, and overall assessment
    - Highlights: 5 specific positive points, achievements, strengths, successful projects
    - Lowlights: 3 areas of concern, weaknesses, challenges, areas for improvement
    - Key Entities: Extract ONLY the most relevant and specific entities mentioned in THIS interview. Focus on unique companies, specific technologies, frameworks, tools, and methodologies that are actually discussed. Avoid generic terms. Be selective and context-specific.
    - Timeline: 5 chronological sections with realistic timestamps and descriptions

    Transcript:
    {transcript}

    Respond only with valid JSON, no additional text.
    """
    
    messages = [{"role": "user", "content": prompt}]
    response = call_chatgpt_api(api_key, messages)
    
    try:
        # Try to parse as JSON
        import json
        result = json.loads(response.strip())
        return result
    except json.JSONDecodeError:
        # Fallback: parse manually if JSON parsing fails
        logger.warning("Failed to parse JSON response, using fallback parsing")
        return parse_fallback_response(response)

def analyze_transcript_fallback(transcript: str) -> dict:
    """
    Fallback analysis using basic NLP techniques when ChatGPT API fails
    
    Args:
        transcript: Interview transcript text
    
    Returns:
        Dictionary with analysis results
    """
    logger.info("Using fallback analysis - ChatGPT API unavailable")
    
    # Clean and prepare text
    text = transcript.lower()
    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?]+', transcript)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Basic statistics
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Extract key entities (names, companies, technologies)
    key_entities = extract_entities_fallback(text)
    
    # Generate summary based on content analysis
    summary = generate_summary_fallback(sentences, word_count)
    
    # Extract highlights and lowlights based on sentiment indicators
    highlights, lowlights = extract_sentiment_fallback(sentences)
    
    # Generate timeline based on content structure
    timeline = generate_timeline_fallback(sentences, word_count)
    
    return {
        "summary": summary,
        "highlights": highlights,
        "lowlights": lowlights,
        "key_entities": key_entities,
        "timeline": timeline
    }

def extract_entities_fallback(text: str) -> list:
    """Extract key entities using pattern matching - focus on context-specific entities"""
    entities = []
    
    # Extract specific company names mentioned
    company_patterns = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Tech|Systems|Solutions|Platforms|AI|Software)\b',
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Google|Microsoft|Amazon|Apple|Meta|Netflix|Uber|Airbnb|Oracle|Salesforce)\b',
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:IBM|Adobe|Intel|AMD|NVIDIA|Cisco|VMware|SAP|Workday)\b'
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if match.lower() not in ['the company', 'our company', 'this company']:
                entities.append(match)
    
    # Extract specific product/platform names
    product_patterns = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:BuildFlow|ChatGPT|Slack|Zoom|Teams|Notion|Figma|Miro|Asana|Trello)\b',
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|GitHub|GitLab)\b'
    ]
    
    for pattern in product_patterns:
        matches = re.findall(pattern, text)
        entities.extend(matches)
    
    # Extract specific frameworks/methodologies mentioned
    framework_patterns = [
        r'\b(?:RICE|MoSCoW|Agile|Scrum|Kanban|Waterfall|DevOps|CI/CD|TDD|BDD)\b',
        r'\b(?:React|Angular|Vue|Node\.js|Django|Flask|FastAPI|Spring|Laravel|Rails)\b'
    ]
    
    for pattern in framework_patterns:
        matches = re.findall(pattern, text)
        entities.extend(matches)
    
    # Extract specific technologies mentioned
    tech_patterns = [
        r'\b(?:Python|JavaScript|Java|C\+\+|SQL|MongoDB|PostgreSQL|Redis|Kafka|Docker|Kubernetes)\b',
        r'\b(?:Machine Learning|AI|Data Science|Analytics|A/B Testing|User Research|Qualitative Research)\b'
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, text)
        entities.extend(matches)
    
    found_tech = []
    for tech in tech_keywords:
        if tech.lower() in text.lower():
            # Use proper case from the text
            pattern = re.compile(re.escape(tech), re.IGNORECASE)
            match = pattern.search(text)
            if match:
                found_tech.append(match.group())
    
    entities.extend(found_tech[:5])
    
    # Extract company names (more specific patterns)
    company_patterns = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Tech|Systems|Solutions)\b',
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Google|Microsoft|Amazon|Apple|Meta|Netflix|Uber|Airbnb)\b',
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:IBM|Oracle|Salesforce|Adobe|Intel|AMD|NVIDIA)\b'
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text)
        entities.extend(matches[:2])
    
    # Extract role/title keywords
    role_keywords = [
        'engineer', 'developer', 'manager', 'lead', 'architect', 'analyst', 'designer',
        'product manager', 'project manager', 'scrum master', 'devops', 'sre', 'qa',
        'frontend', 'backend', 'fullstack', 'data engineer', 'ml engineer'
    ]
    
    found_roles = []
    for role in role_keywords:
        if role.lower() in text.lower():
            pattern = re.compile(re.escape(role), re.IGNORECASE)
            match = pattern.search(text)
            if match:
                found_roles.append(match.group())
    
    entities.extend(found_roles[:3])
    
    # Extract framework/library names
    framework_keywords = [
        'spring', 'django', 'flask', 'express', 'fastapi', 'laravel', 'rails',
        'bootstrap', 'tailwind', 'material-ui', 'ant design', 'jquery', 'lodash',
        'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn', 'jupyter'
    ]
    
    found_frameworks = []
    for framework in framework_keywords:
        if framework.lower() in text.lower():
            pattern = re.compile(re.escape(framework), re.IGNORECASE)
            match = pattern.search(text)
            if match:
                found_frameworks.append(match.group())
    
    entities.extend(found_frameworks[:3])
    
    # Remove duplicates and filter out common words and names
    common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    name_patterns = [r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', r'\b[A-Z][a-z]+\b']  # Common name patterns
    
    filtered_entities = []
    
    for entity in entities:
        # Skip if it's a common word or looks like a name
        if (entity.lower() not in common_words and 
            len(entity) > 2 and 
            not any(re.match(pattern, entity) for pattern in name_patterns)):
            filtered_entities.append(entity)
    
    return list(set(filtered_entities))[:8]  # Remove duplicates and limit

def generate_summary_fallback(sentences: list, word_count: int) -> str:
    """Generate summary based on content analysis"""
    if not sentences:
        return "Interview transcript analyzed successfully."
    
    # Filter out timestamp patterns and speaker indicators
    clean_sentences = []
    for sentence in sentences:
        # Remove timestamp patterns like [00:00] or [00:00:00]
        clean_sentence = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', sentence)
        # Remove speaker patterns like "Rachel:" or "Lokesh:"
        clean_sentence = re.sub(r'^[A-Za-z]+:\s*', '', clean_sentence)
        clean_sentence = clean_sentence.strip()
        
        if len(clean_sentence) > 15 and not clean_sentence.startswith('['):
            clean_sentences.append(clean_sentence)
    
    if not clean_sentences:
        return f"Interview transcript analyzed with {word_count} words across {len(sentences)} sentences."
    
    # Enhanced scoring system for better summary
    key_indicators = {
        'high_priority': ['experience', 'background', 'project', 'team', 'leadership', 'technical', 'challenge', 'solution', 'goal', 'achievement', 'responsibility', 'management', 'development', 'implementation'],
        'medium_priority': ['work', 'company', 'role', 'position', 'skills', 'technology', 'process', 'strategy', 'planning', 'execution', 'delivery', 'results', 'impact', 'success'],
        'low_priority': ['time', 'year', 'month', 'day', 'start', 'end', 'begin', 'finish', 'complete', 'done', 'working', 'doing', 'making']
    }
    
    scored_sentences = []
    
    for sentence in clean_sentences:
        score = 0
        sentence_lower = sentence.lower()
        
        # Score based on priority indicators
        for indicator in key_indicators['high_priority']:
            if indicator in sentence_lower:
                score += 3
        
        for indicator in key_indicators['medium_priority']:
            if indicator in sentence_lower:
                score += 2
        
        for indicator in key_indicators['low_priority']:
            if indicator in sentence_lower:
                score += 1
        
        # Bonus for longer sentences (more content)
        if len(sentence) > 50:
            score += 2
        elif len(sentence) > 30:
            score += 1
        
        # Penalty for very short sentences
        if len(sentence) < 20:
            score -= 1
        
        # Bonus for sentences with numbers (often indicate achievements)
        if re.search(r'\d+', sentence):
            score += 1
        
        scored_sentences.append((score, sentence))
    
    # Sort by score and take top sentences
    scored_sentences.sort(reverse=True)
    top_sentences = [sentence for score, sentence in scored_sentences[:5] if score > 0]
    
    if top_sentences:
        # Create a more coherent summary
        if len(top_sentences) >= 2:
            summary = f"{top_sentences[0]} {top_sentences[1]}"
        else:
            summary = top_sentences[0]
        
        if len(summary) > 400:
            summary = summary[:400] + "..."
        return summary
    else:
        # Fallback to first substantial sentence
        for sentence in clean_sentences:
            if len(sentence) > 30:
                return sentence[:250] + "..." if len(sentence) > 250 else sentence
        
        return f"Interview transcript analyzed with {word_count} words covering various topics and experiences."

def extract_sentiment_fallback(sentences: list) -> tuple[list, list]:
    """Extract highlights and lowlights based on sentiment indicators"""
    highlights = []
    lowlights = []
    
    # Enhanced positive indicators
    positive_words = [
        'excellent', 'great', 'good', 'strong', 'impressive', 'successful', 'achieved',
        'improved', 'enhanced', 'optimized', 'efficient', 'effective', 'innovative',
        'creative', 'collaborative', 'leadership', 'expertise', 'experience', 'accomplished',
        'delivered', 'led', 'managed', 'developed', 'built', 'created', 'solved', 'resolved',
        'outstanding', 'exceptional', 'remarkable', 'notable', 'significant', 'valuable',
        'proven', 'demonstrated', 'established', 'implemented', 'launched', 'completed',
        'exceeded', 'surpassed', 'boosted', 'increased', 'improved', 'enhanced', 'streamlined',
        'optimized', 'automated', 'scaled', 'expanded', 'grown', 'developed', 'mentored',
        'guided', 'supported', 'facilitated', 'coordinated', 'organized', 'planned'
    ]
    
    # Enhanced negative indicators
    negative_words = [
        'challenge', 'difficult', 'problem', 'issue', 'weakness', 'concern', 'risk',
        'failure', 'error', 'bug', 'slow', 'inefficient', 'conflict', 'stress',
        'pressure', 'deadline', 'complex', 'complicated', 'struggle', 'limitation',
        'constraint', 'obstacle', 'hurdle', 'setback', 'mistake', 'missed',
        'difficulty', 'trouble', 'obstacle', 'barrier', 'blockage', 'impediment',
        'hindrance', 'drawback', 'disadvantage', 'weakness', 'shortcoming', 'flaw',
        'defect', 'fault', 'deficiency', 'lack', 'absence', 'missing', 'incomplete',
        'unfinished', 'delayed', 'overdue', 'behind', 'lagging', 'falling', 'declining',
        'reducing', 'decreasing', 'diminishing', 'worsening', 'deteriorating'
    ]
    
    # Analyze sentences for sentiment
    for sentence in sentences:
        # Clean sentence - remove timestamps and speaker indicators
        clean_sentence = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', sentence)
        clean_sentence = re.sub(r'^[A-Za-z]+:\s*', '', clean_sentence)
        clean_sentence = clean_sentence.strip()
        
        # Skip if sentence is too short or contains only timestamps
        if len(clean_sentence) < 10 or clean_sentence.startswith('['):
            continue
            
        sentence_lower = clean_sentence.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in sentence_lower)
        negative_count = sum(1 for word in negative_words if word in sentence_lower)
        
        # Only include meaningful content
        if positive_count > negative_count and positive_count > 0 and len(clean_sentence) > 20:
            highlights.append(clean_sentence[:120] + "..." if len(clean_sentence) > 120 else clean_sentence)
        elif negative_count > positive_count and negative_count > 0 and len(clean_sentence) > 20:
            lowlights.append(clean_sentence[:120] + "..." if len(clean_sentence) > 120 else clean_sentence)
    
    # Ensure we have some results with meaningful content
    if not highlights:
        highlights = [
            "Strong technical background and experience demonstrated",
            "Good communication and collaboration skills shown",
            "Proven track record of project delivery and leadership"
        ]
    if not lowlights:
        lowlights = [
            "Some areas for improvement identified in technical depth",
            "Challenges discussed regarding team dynamics and deadlines",
            "Opportunities for growth in advanced technical skills"
        ]
    
    return highlights[:5], lowlights[:5]

def generate_timeline_fallback(sentences: list, word_count: int) -> list:
    """Generate timeline based on content structure"""
    timeline = []
    
    if not sentences:
        return [{"timestamp": "00:00:00", "topic": "Interview", "description": "Analysis completed"}]
    
    # Estimate interview duration (roughly 150 words per minute)
    estimated_minutes = max(1, word_count // 150)
    
    # Create timeline sections
    sections = [
        {"timestamp": "00:00:00", "topic": "Introduction", "description": "Interview begins and introductions"},
        {"timestamp": f"00:0{estimated_minutes//3:02d}:00", "topic": "Background Discussion", "description": "Candidate background and experience"},
        {"timestamp": f"00:0{2*estimated_minutes//3:02d}:00", "topic": "Technical Discussion", "description": "Technical questions and problem solving"},
        {"timestamp": f"00:{estimated_minutes:02d}:00", "topic": "Conclusion", "description": "Interview wrap-up and questions"}
    ]
    
    return sections

def parse_fallback_response(response: str) -> dict:
    """Fallback parsing if JSON response fails"""
    lines = response.split('\n')
    summary = ""
    highlights = []
    lowlights = []
    key_entities = []
    timeline = []
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if 'summary' in line.lower() and ':' in line:
            summary = line.split(':', 1)[1].strip().strip('"')
        elif 'highlight' in line.lower():
            current_section = 'highlights'
        elif 'lowlight' in line.lower():
            current_section = 'lowlights'
        elif 'entity' in line.lower():
            current_section = 'entities'
        elif 'timeline' in line.lower():
            current_section = 'timeline'
        elif line.startswith('-') or line.startswith('•'):
            content = line[1:].strip().strip('"')
            if current_section == 'highlights':
                highlights.append(content)
            elif current_section == 'lowlights':
                lowlights.append(content)
            elif current_section == 'entities':
                key_entities.append(content)
        elif 'timestamp' in line.lower() and 'topic' in line.lower():
            # Simple timeline parsing
            if '00:00:00' in line:
                timeline.append({
                    "timestamp": "00:00:00",
                    "topic": "Introduction",
                    "description": "Interview begins"
                })
            elif '00:02:30' in line:
                timeline.append({
                    "timestamp": "00:02:30",
                    "topic": "Technical Discussion",
                    "description": "Technical questions and answers"
                })
    
    return {
        "summary": summary or "Interview analysis completed",
        "highlights": highlights[:5],
        "lowlights": lowlights[:5],
        "key_entities": key_entities[:5],
        "timeline": timeline or [
            {
                "timestamp": "00:00:00",
                "topic": "Interview",
                "description": "Analysis completed"
            }
        ]
    }



def check_title_exists(title: str) -> bool:
    """
    Check if a title already exists in the database
    
    Args:
        title: Interview title to check
    
    Returns:
        True if title exists, False otherwise
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM interviews WHERE title = ?", (title,))
        count = cursor.fetchone()[0]
        return count > 0

def save_interview_to_db(interview_id: str, title: str, transcript: str, analysis_result: dict, upload_method: str = "manual", original_filename: str = None):
    """
    Save interview and analysis results to database
    
    Args:
        interview_id: Unique interview ID
        title: Interview title
        transcript: Original transcript
        analysis_result: Analysis results dictionary
        upload_method: Method used to upload ('manual' or 'file')
        original_filename: Original filename if uploaded via file
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO interviews (
                id, title, transcript, summary, highlights, lowlights, 
                key_entities, timeline, analysis_metadata, upload_method, original_filename, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            interview_id,
            title,
            transcript,
            analysis_result['summary'],
            json.dumps(analysis_result['highlights']),
            json.dumps(analysis_result['lowlights']),
            json.dumps(analysis_result['key_entities']),
            json.dumps([item.dict() if hasattr(item, 'dict') else item for item in analysis_result['timeline']]),
            json.dumps(analysis_result['analysis_metadata']),
            upload_method,
            original_filename,
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        conn.commit()

def search_interviews_by_title(search_term: str) -> List[Dict]:
    """
    Search interviews by title
    
    Args:
        search_term: Search term to match against titles
    
    Returns:
        List of matching interviews
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, title, created_at, summary, upload_method, original_filename
            FROM interviews 
            WHERE title LIKE ? 
            ORDER BY created_at DESC
        ''', (f'%{search_term}%',))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'interview_id': row['id'],
                'title': row['title'],
                'created_at': row['created_at'],
                'summary': row['summary'],
                'upload_method': row['upload_method'],
                'original_filename': row['original_filename']
            })
        return results

def get_interview_by_id(interview_id: str) -> Optional[Dict]:
    """
    Get interview by ID
    
    Args:
        interview_id: Unique interview ID
    
    Returns:
        Interview data or None if not found
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM interviews WHERE id = ?
        ''', (interview_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                'interview_id': row['id'],
                'title': row['title'],
                'transcript': row['transcript'],
                'summary': row['summary'],
                'highlights': json.loads(row['highlights']) if row['highlights'] else [],
                'lowlights': json.loads(row['lowlights']) if row['lowlights'] else [],
                'key_entities': json.loads(row['key_entities']) if row['key_entities'] else [],
                'timeline': [TimelineItem(**item) for item in json.loads(row['timeline'])] if row['timeline'] else [],
                'analysis_metadata': json.loads(row['analysis_metadata']) if row['analysis_metadata'] else {},
                'upload_method': row['upload_method'],
                'original_filename': row['original_filename'],
                'created_at': row['created_at']
            }
        return None

@app.post("/api/analyze-transcript", response_model=AnalysisResponse)
async def analyze_transcript(request: TranscriptRequest):
    """
    Analyze interview transcript and return comprehensive analysis
    
    Args:
        request: TranscriptRequest containing title, transcript and API key
    
    Returns:
        AnalysisResponse with analysis results and interview ID
    """
    try:
        logger.info(f"Starting transcript analysis for title: {request.title}")
        
        # Check if API key is available
        if not OPENAI_API_KEY:
            logger.warning("No OpenAI API key configured - using fallback analysis")
        else:
            api_key_preview = OPENAI_API_KEY[:10] + "..." if len(OPENAI_API_KEY) > 10 else OPENAI_API_KEY
            logger.info(f"Using API key: {api_key_preview} (length: {len(OPENAI_API_KEY)})")
        
        # Check if title already exists
        if check_title_exists(request.title):
            raise HTTPException(
                status_code=400, 
                detail=f"Interview title '{request.title}' already exists. Please choose a different title."
            )
        
        # Generate unique interview ID
        interview_id = str(uuid.uuid4())
        
        # Try ChatGPT API first, fallback to local analysis if it fails
        try:
            if not OPENAI_API_KEY:
                raise Exception("No OpenAI API key configured")
                
            logger.info("Attempting ChatGPT API analysis...")
            analysis_result = analyze_transcript_comprehensive(OPENAI_API_KEY, request.transcript)
            logger.info("ChatGPT API analysis successful!")
            
            summary = analysis_result.get('summary', 'Analysis completed')
            highlights = analysis_result.get('highlights', [])
            lowlights = analysis_result.get('lowlights', [])
            key_entities = analysis_result.get('key_entities', [])
            
            # Convert timeline to TimelineItem objects
            timeline_data = analysis_result.get('timeline', [])
            timeline = []
            for item in timeline_data:
                timeline.append(TimelineItem(
                    timestamp=item.get('timestamp', '00:00:00'),
                    topic=item.get('topic', 'Section'),
                    description=item.get('description', 'Description')
                ))
                
        except Exception as e:
            # Use fallback analysis when ChatGPT API fails
            logger.error(f"ChatGPT API failed with error: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Check if it's a rate limit issue
            if "rate limit" in str(e).lower() or "429" in str(e):
                logger.warning("Rate limit exceeded - using fallback analysis. Try again in a few minutes for ChatGPT analysis.")
            else:
                logger.warning("ChatGPT API unavailable - using fallback analysis...")
            
            fallback_result = analyze_transcript_fallback(request.transcript)
            
            summary = fallback_result.get('summary', 'Analysis completed using fallback method')
            highlights = fallback_result.get('highlights', [])
            lowlights = fallback_result.get('lowlights', [])
            key_entities = fallback_result.get('key_entities', [])
            
            # Convert timeline to TimelineItem objects
            timeline_data = fallback_result.get('timeline', [])
            timeline = []
            for item in timeline_data:
                timeline.append(TimelineItem(
                    timestamp=item.get('timestamp', '00:00:00'),
                    topic=item.get('topic', 'Section'),
                    description=item.get('description', 'Description')
                ))
        
        # Track analysis start time
        analysis_start_time = time.time()
        
        # Create metadata
        analysis_metadata = {
            "analysis_timestamp": datetime.now().isoformat(),
            "transcript_length": len(request.transcript),
            "word_count": len(request.transcript.split()),
            "model_used": DEFAULT_MODEL,
            "analysis_components": ["summary", "highlights", "lowlights", "key_entities", "timeline"],
            "analysis_method": "fallback" if "fallback" in locals() else "chatgpt",
            "analysis_time_seconds": round(time.time() - analysis_start_time, 2)
        }
        
        # Save to database
        analysis_result = {
            'summary': summary,
            'highlights': highlights,
            'lowlights': lowlights,
            'key_entities': key_entities,
            'timeline': timeline,
            'analysis_metadata': analysis_metadata
        }
        
        save_interview_to_db(interview_id, request.title, request.transcript, analysis_result, "manual")
        
        logger.info(f"Transcript analysis completed successfully for ID: {interview_id}")
        
        return AnalysisResponse(
            interview_id=interview_id,
            title=request.title,
            summary=summary,
            highlights=highlights,
            lowlights=lowlights,
            key_entities=key_entities,
            timeline=timeline,
            analysis_metadata=analysis_metadata,
            upload_method="manual",
            original_filename=None,
            created_at=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during transcript analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze-file", response_model=AnalysisResponse)
async def analyze_file(
    file: UploadFile = File(...),
    title: str = Form(...)
):
    """
    Upload file and analyze transcript in one step
    
    Args:
        file: Uploaded file (PDF, TXT, etc.)
        title: Interview title
    
    Returns:
        AnalysisResponse with analysis results
    """
    try:
        logger.info(f"Processing and analyzing file: {file.filename} with title: {title}")
        
        # Check if title already exists
        if check_title_exists(title):
            raise HTTPException(status_code=400, detail=f"Interview title '{title}' already exists. Please use a unique title.")
        
        # Check file size (limit to 10MB)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        
        # Read file content
        content = await file.read()
        transcript = ""
        
        # Process based on file type
        if file.content_type == "application/pdf":
            # Handle PDF files
            try:
                pdf_file = io.BytesIO(content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                text_content = ""
                for page_num in range(len(pdf_reader.pages)):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        
                        # Clean and format the extracted text
                        if page_text:
                            cleaned_text = clean_extracted_text(page_text)
                            text_content += cleaned_text + "\n\n"
                    except Exception as page_error:
                        logger.warning(f"Error reading page {page_num + 1}: {str(page_error)}")
                        text_content += f"[Page {page_num + 1} - Text extraction failed]\n\n"
                
                if not text_content.strip():
                    raise HTTPException(status_code=400, detail="No text content could be extracted from the PDF. The file might be scanned or image-based.")
                
                # Final cleanup of the entire text
                transcript = clean_extracted_text(text_content)
                
            except Exception as pdf_error:
                logger.error(f"PDF processing error: {str(pdf_error)}")
                raise HTTPException(status_code=400, detail=f"PDF processing failed: {str(pdf_error)}")
        
        elif file.content_type in ["text/plain", "text/markdown"]:
            # Handle text files
            try:
                transcript = content.decode('utf-8')
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8 encoding.")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Supported types: PDF, TXT, MD")
        
        if not transcript.strip():
            raise HTTPException(status_code=400, detail="No text content found in the file.")
        
        # Generate unique ID
        interview_id = str(uuid.uuid4())
        
        # Analyze the transcript
        analysis_start_time = time.time()
        
        try:
            # Try ChatGPT analysis first
            analysis_result = analyze_transcript_comprehensive(OPENAI_API_KEY, transcript)
            analysis_method = "chatgpt"
        except Exception as chatgpt_error:
            logger.warning(f"ChatGPT analysis failed, using fallback: {str(chatgpt_error)}")
            analysis_result = analyze_transcript_fallback(transcript)
            analysis_method = "fallback"
        
        # Calculate analysis time
        analysis_time_seconds = round(time.time() - analysis_start_time, 2)
        
        # Extract results
        summary = analysis_result['summary']
        highlights = analysis_result['highlights']
        lowlights = analysis_result['lowlights']
        key_entities = analysis_result['key_entities']
        timeline = analysis_result['timeline']
        
        # Add metadata
        analysis_metadata = {
            'analysis_method': analysis_method,
            'analysis_time_seconds': analysis_time_seconds,
            'word_count': len(transcript.split())
        }
        
        # Add metadata to analysis result
        analysis_result['analysis_metadata'] = analysis_metadata
        
        # Save to database with file upload method
        save_interview_to_db(
            interview_id, 
            title, 
            transcript, 
            analysis_result, 
            "file", 
            file.filename
        )
        
        logger.info(f"File analysis completed successfully for ID: {interview_id}")
        
        return AnalysisResponse(
            interview_id=interview_id,
            title=title,
            summary=summary,
            highlights=highlights,
            lowlights=lowlights,
            key_entities=key_entities,
            timeline=timeline,
            analysis_metadata=analysis_metadata,
            upload_method="file",
            original_filename=file.filename,
            created_at=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during file analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File analysis failed: {str(e)}")

@app.get("/api/search-interviews", response_model=SearchResponse)
async def search_interviews(search_term: str = Query(..., description="Search term for interview titles")):
    """
    Search interviews by title
    
    Args:
        search_term: Search term to match against interview titles
    
    Returns:
        SearchResponse with matching interviews
    """
    try:
        logger.info(f"Searching interviews with term: {search_term}")
        
        results = search_interviews_by_title(search_term)
        
        interview_list = [
            InterviewListItem(
                interview_id=item['interview_id'],
                title=item['title'],
                created_at=item['created_at'],
                summary=item['summary'],
                upload_method=item['upload_method'],
                original_filename=item['original_filename']
            ) for item in results
        ]
        
        return SearchResponse(
            interviews=interview_list,
            total_count=len(interview_list)
        )
    
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/interview/{interview_id}", response_model=AnalysisResponse)
async def get_interview(interview_id: str):
    """
    Get interview by ID
    
    Args:
        interview_id: Unique interview ID
    
    Returns:
        AnalysisResponse with interview data
    """
    try:
        logger.info(f"Retrieving interview with ID: {interview_id}")
        
        interview_data = get_interview_by_id(interview_id)
        
        if not interview_data:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        return AnalysisResponse(**interview_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving interview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@app.get("/api/interviews", response_model=SearchResponse)
async def list_all_interviews():
    """
    List all interviews
    
    Returns:
        SearchResponse with all interviews
    """
    try:
        logger.info("Retrieving all interviews")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, title, created_at, summary, upload_method, original_filename
                FROM interviews 
                ORDER BY created_at DESC
            ''')
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'interview_id': row['id'],
                    'title': row['title'],
                    'created_at': row['created_at'],
                    'summary': row['summary'],
                    'upload_method': row['upload_method'],
                    'original_filename': row['original_filename']
                })
        
        interview_list = [
            InterviewListItem(
                interview_id=item['interview_id'],
                title=item['title'],
                created_at=item['created_at'],
                summary=item['summary'],
                upload_method=item['upload_method'],
                original_filename=item['original_filename']
            ) for item in results
        ]
        
        return SearchResponse(
            interviews=interview_list,
            total_count=len(interview_list)
        )
    
    except Exception as e:
        logger.error(f"Error listing interviews: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Listing failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process file to extract text content
    
    Args:
        file: Uploaded file (PDF, TXT, etc.)
    
    Returns:
        Dictionary with extracted text and file info
    """
    try:
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Check file size (limit to 10MB)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        
        # Read file content
        content = await file.read()
        
        # Process based on file type
        if file.content_type == "application/pdf":
            # Handle PDF files
            try:
                pdf_file = io.BytesIO(content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                text_content = ""
                for page_num in range(len(pdf_reader.pages)):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        
                        # Clean and format the extracted text
                        if page_text:
                            cleaned_text = clean_extracted_text(page_text)
                            text_content += cleaned_text + "\n\n"
                    except Exception as page_error:
                        logger.warning(f"Error reading page {page_num + 1}: {str(page_error)}")
                        text_content += f"[Page {page_num + 1} - Text extraction failed]\n\n"
                
                if not text_content.strip():
                    raise HTTPException(status_code=400, detail="No text content could be extracted from the PDF. The file might be scanned or image-based.")
                
                # Final cleanup of the entire text
                final_text = clean_extracted_text(text_content)
                
                return {
                    "status": "success",
                    "filename": file.filename,
                    "content": final_text,
                    "file_type": "pdf",
                    "pages": len(pdf_reader.pages),
                    "message": f"Successfully extracted text from {len(pdf_reader.pages)} pages"
                }
                
            except Exception as pdf_error:
                logger.error(f"PDF processing error: {str(pdf_error)}")
                raise HTTPException(status_code=400, detail=f"PDF processing failed: {str(pdf_error)}")
        
        elif file.content_type in ["text/plain", "text/markdown"]:
            # Handle text files
            try:
                text_content = content.decode('utf-8')
                return {
                    "status": "success",
                    "filename": file.filename,
                    "content": text_content,
                    "file_type": "text",
                    "message": "Successfully read text file"
                }
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8 encoding.")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Supported types: PDF, TXT, MD")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.post("/api/test-api-key")
async def test_api_key():
    """Test OpenAI API key endpoint"""
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=400, detail="No OpenAI API key configured in backend")
        
        # Simple test with ChatGPT
        test_messages = [{"role": "user", "content": "Say 'Hello, API key is working!'"}]
        response = call_chatgpt_api(OPENAI_API_KEY, test_messages)
        
        return {
            "status": "success",
            "message": "API key is valid",
            "response": response
        }
    except Exception as e:
        logger.error(f"API key test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API key test failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Interview Transcript Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/analyze-transcript",
            "search": "/api/search-interviews",
            "get_interview": "/api/interview/{interview_id}",
            "list_all": "/api/interviews",
            "health": "/api/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (for production) or use default
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run("main:app", host=host, port=port, reload=False)
