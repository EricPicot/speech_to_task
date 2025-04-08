# Speech to Task Tracker

A Python-based application that converts speech to structured task data and syncs with Notion, built with Gradio and OpenRouter.

## Features

- 🎤 Voice input processing with automatic voice detection
- 📝 Speech-to-text conversion
- 🔍 Task extraction and structuring
- 📊 Notion integration
- ⏱️ Time tracking
- 🌐 Modern web UI built with Gradio

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

3. Run the application:
```bash
# Standard version
python src/main.py

# Advanced version with voice activity detection
python src/advanced_main.py
```

## Docker Setup

You can also run the application with Docker:

```bash
# Build and run the Docker container
docker-compose up -d

# To run only the standard version
docker-compose up app

# To run only the advanced version
docker-compose up advanced
```

Access the application:
- Standard version: http://localhost:8000
- Advanced version: http://localhost:8001

## Project Structure

```
speech_to_task/
├── src/
│   ├── main.py           # Main application with Gradio UI
│   ├── advanced_main.py  # Advanced version with voice activity detection
│   ├── speech_to_text.py # Speech-to-text conversion
│   ├── task_parser.py    # Task extraction and structuring
│   └── notion_sync.py    # Notion integration
├── tests/                # Test files
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
└── README.md             # Project documentation
```

## Voice Recording Options

The application provides two ways to record your voice:

1. **Manual Recording**: Click the record button, speak, then click stop.
2. **Automatic Voice Detection**: The system automatically detects when you start and stop speaking (available in the advanced version).

## Using in Different Languages

The system is designed to work with both English and French. The speech-to-text system (OpenRouter) handles multiple languages, and the task extraction is designed to work with either language.

## Requirements

- Python 3.8+
- OpenRouter API key
- Notion API key and database ID

## License

MIT
