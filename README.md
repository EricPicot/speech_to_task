# Speech to Task Tracker

A Python-based application that converts speech to structured task data and syncs with Notion, built with Gradio and OpenRouter.

## Features

- 🎤 Voice input processing with automatic voice detection
- 📝 Speech-to-text conversion
- 🔍 Task extraction and structuring
- 📊 Notion integration
- ⏱️ Time tracking
- 🌐 Modern web UI built with Gradio
- 🔄 Microservices architecture (backend API + frontend)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
brew install flac
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenRouter API key, Notion API key, and other settings
```

3. Create necessary directories:
```bash
mkdir -p audio_input transcripts logs
chmod 777 audio_input transcripts logs
```

4. Run the application:
```bash
# Run backend API (FastAPI)
python src/backend_api.py

# Run frontend (Gradio)
python src/frontend_gradio.py
```

## Docker Setup

You can run the application with Docker using a microservices architecture:

```bash
# Build and run both backend and frontend
docker-compose up -d

# To run only the backend
docker-compose up backend

# To run only the frontend
docker-compose up frontend

# Check logs
docker-compose logs -f
```

Access the application:
- Frontend (Gradio UI): http://localhost:8000
- Backend API: http://localhost:8080
- API Documentation: http://localhost:8080/docs

## Microservices Architecture

The application is divided into two main components:

1. **Backend (FastAPI)**: 
   - Handles audio processing and task extraction
   - Provides RESTful API endpoints
   - Communicates with OpenRouter for task extraction
   - Maintains audio and transcript storage

2. **Frontend (Gradio)**:
   - Provides user interface
   - Records audio and collects text input
   - Communicates with backend via HTTP requests
   - Displays results in a user-friendly format

## Project Structure

```
speech_to_task/
├── src/
│   ├── frontend_gradio.py  # Gradio UI frontend
│   ├── backend_api.py      # FastAPI backend server
│   ├── task_extractor.py   # Simple task extraction
│   ├── task_agent_extractor.py # Advanced task extraction with LangChain agent
│   └── notion_sync.py      # Notion integration
├── docker/
│   ├── backend.Dockerfile  # Docker configuration for backend
│   └── frontend.Dockerfile # Docker configuration for frontend
├── tests/                  # Test files
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Docker Compose configuration
└── README.md               # Project documentation
```

## Task Extraction Options

The application provides two methods for extracting tasks:

1. **Simple Extractor**: Faster but less robust, extracts basic task details.
2. **Agent Extractor**: More robust, uses LangChain agents to understand context, resolve temporal references, and handle more complex descriptions.

## Troubleshooting

### Connection Issues
- **Frontend can't connect to backend**: 
  - Check if the backend is running and accessible
  - Verify the BACKEND_URL environment variable is correctly set
  - In Docker, ensure the network is correctly configured
  - Try running `curl http://localhost:8080/` to check backend availability

### Audio Processing Issues
- **Audio not recognized**: 
  - Check your microphone is working correctly
  - Ensure you have flac installed: `brew install flac` on macOS or `apt-get install flac` on Linux
  - Verify the audio is saved correctly by checking the audio_input directory

### Docker Issues
- **Containers not starting**:
  - Check logs with `docker-compose logs -f`
  - Ensure all required environment variables are set
  - Make sure ports 8000 and 8080 are not already in use
  - Run `docker-compose down -v` and then `docker-compose up -d` to rebuild from scratch

### OpenRouter API Issues
- **Task extraction not working**:
  - Verify your OpenRouter API key is valid and correctly set
  - Check the OpenRouter dashboard for quota or rate limiting issues
  - Look at backend logs for detailed error messages

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OPENROUTER_API_KEY | Your OpenRouter API key | Required |
| NOTION_API_KEY | Your Notion API key | Optional |
| NOTION_DATABASE_ID | Your Notion database ID | Optional |
| BACKEND_URL | URL of the backend API | http://localhost:8080 |
| DEBUG | Enable debug mode | 0 (disabled) |

## Requirements

- Python 3.8+
- OpenRouter API key
- Notion API key and database ID (optional)

## License

MIT
