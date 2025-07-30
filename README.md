# Interview Transcript Analyzer - One-Stop Platform

A comprehensive AI-powered interview transcript analysis tool that provides intelligent insights, timeline summaries, and key entity extraction using ChatGPT integration.

## 🚀 Features

### Core Analysis Capabilities
- **Intelligent Summary**: Timeline-ordered summary with timestamps
- **Highlights & Lowlights**: AI-powered sentiment analysis for hiring decisions
- **Named Entity Extraction**: People, companies, technologies, and key terms
- **Chronological Timeline**: Detailed breakdown of interview progression
- **Metrics**: Word count and duration estimates

### Technical Features
- **ChatGPT Integration**: Advanced AI analysis using OpenAI's GPT models
- **Real-time Processing**: Fast analysis with intelligent fallbacks
- **File Upload Support**: Multiple format support (.txt, .md, .doc, .docx)
- **Responsive UI**: Modern, intuitive interface
- **RESTful API**: Complete backend API for integration

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- OpenAI API key (for ChatGPT integration)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd finalRoundAI
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Set up environment variables (if needed)
cp env.example .env.local
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Application Settings
APP_NAME=Interview Transcript Analyzer
APP_VERSION=1.0.0
DEBUG=True
ENVIRONMENT=development

# Server Settings
HOST=0.0.0.0
PORT=8000
RELOAD=True

# ChatGPT API Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_API_URL=https://api.openai.com/v1/chat/completions

# CORS Settings
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### Getting OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an account or sign in
3. Generate a new API key
4. Add the key to your `.env` file

## 🚀 Running the Application

### 1. Start the Backend

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

The backend will start on `http://localhost:8000`

### 2. Start the Frontend

```bash
cd frontend
npm run dev
```

The frontend will start on `http://localhost:3000`

### 3. Access the Application

Open your browser and navigate to `http://localhost:3000`

## 📖 Usage

### Web Interface

1. **Upload Transcript**: Either paste text or upload a file
2. **Set Title**: Give your interview a descriptive title
3. **Analyze**: Click "Analyze Transcript" to process
4. **Review Results**: View comprehensive analysis including:
   - Timeline summary with timestamps
   - Highlights and lowlights
   - Named entities
   - Metrics and insights

### API Usage

#### Analyze Transcript
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "00:00:10 Interviewer: Welcome...",
    "title": "Software Engineer Interview"
  }'
```

#### Get Health Status
```bash
curl http://localhost:8000/health
```

#### Get Demo Transcript
```bash
curl http://localhost:8000/demo
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
cd backend
source venv/bin/activate
python test_analysis.py
```

This will test:
- Health endpoints
- Analysis functionality
- API responses
- ChatGPT integration

## 📊 Sample Output

### Timeline Summary
```
- 00:00:10   introduction   Welcome and background discussion
- 00:02:30   technical questions   React and Node.js experience
- 00:05:00   project discussion   Google analytics dashboard project
- 00:08:00   behavioral questions   Team conflict resolution
- 00:11:00   salary discussion   Compensation expectations
- 00:13:00   closing   Questions and next steps
```

### Analysis Results
```json
{
  "summary": "Timeline-ordered summary...",
  "highlights": [
    "Strong technical skills demonstrated",
    "Excellent project leadership experience",
    "Good communication abilities"
  ],
  "lowlights": [
    "Some concerns about direct feedback",
    "Limited distributed systems experience"
  ],
  "named_entities": [
    "React", "Node.js", "Python", "Google", "FastAPI"
  ],
  "timeline": [...],
  "word_count": 209,
  "duration_estimate": "1 minutes"
}
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message and API info |
| `/health` | GET | Health check and AI status |
| `/analyze` | POST | Analyze interview transcript |
| `/demo` | GET | Get sample transcript |
| `/transcripts` | GET | List all analyzed transcripts |
| `/transcripts/{id}` | GET | Get specific transcript |

## 🏗️ Architecture

### Backend (FastAPI)
- **FastAPI**: Modern Python web framework
- **Pydantic**: Data validation and serialization
- **OpenAI API**: ChatGPT integration for intelligent analysis
- **Uvicorn**: ASGI server

### Frontend (Next.js)
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Modern styling
- **Responsive Design**: Mobile-friendly interface

## 🔍 Analysis Features

### Intelligent Summary
- Timeline-ordered format with timestamps
- Concise descriptions of key discussion points
- Chronological progression tracking

### Highlights & Lowlights
- AI-powered sentiment analysis
- Specific examples and achievements
- Actionable insights for hiring decisions

### Named Entity Extraction
- People names and roles
- Company names and technologies
- Programming languages and frameworks
- Tools and platforms

### Timeline Analysis
- Detailed chronological breakdown
- Topic identification and categorization
- Duration estimates and pacing analysis

## 🚀 Deployment

### Backend Deployment
```bash
# Production setup
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Frontend Deployment
```bash
# Build for production
npm run build
npm start
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

## 🔄 Updates

### Recent Improvements
- Enhanced ChatGPT prompts for better analysis
- Improved timeline formatting
- Better error handling and fallbacks
- Comprehensive test suite
- Updated documentation

### Roadmap
- Database integration for persistent storage
- User authentication and management
- Advanced analytics and reporting
- Export functionality (PDF, CSV)
- Real-time collaboration features 