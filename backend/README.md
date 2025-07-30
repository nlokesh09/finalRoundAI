# Final Round AI - Backend

This is the FastAPI backend for the Final Round AI project.

## Setup

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Development Mode
```bash
python main.py
```

### Using Uvicorn directly
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, you can access:

- **Interactive API docs (Swagger UI):** http://localhost:8000/docs
- **Alternative API docs (ReDoc):** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

## Available Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check
- `GET /items` - Get all items
- `POST /items` - Create a new item
- `GET /items/{item_id}` - Get a specific item
- `GET /users` - Get all users
- `POST /users` - Create a new user

## Environment Variables

Create a `.env` file in the backend directory for environment-specific configurations:

```env
DATABASE_URL=sqlite:///./app.db
SECRET_KEY=your-secret-key-here
DEBUG=True
```

## Project Structure

```
backend/
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── venv/               # Virtual environment
└── .env                # Environment variables (create this)
```

## Next Steps

1. Add database integration (SQLAlchemy)
2. Implement authentication and authorization
3. Add more API endpoints as needed
4. Set up proper error handling and logging
5. Add tests using pytest 