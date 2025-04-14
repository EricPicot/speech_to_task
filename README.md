# Speech to Task Tracker

A Python-based application that converts speech to structured task data and syncs with Notion, built with Gradio and OpenRouter.

## Features

- 🎤 Voice input processing with automatic voice detection
- 📝 Speech-to-text conversion
- 🔍 Task extraction and structuring with LangGraph orchestration
- 📊 Notion integration
- ⏱️ Time tracking
- 🌐 Modern web UI built with Gradio
- 🤖 Parallel task processing with LangGraph agents

## Architecture

The application follows a pipeline architecture with three main stages:

1. **Speech Processing Stage**:
   - Audio capture through Gradio's audio interface
   - Real-time voice activity detection
   - Audio preprocessing with FLAC codec
   - Speech-to-text conversion using SpeechRecognition
   - Transcript storage and validation

2. **Task Extraction Stage** (LangGraph Orchestration):
   - **Coordinator Agent**: 
     - Analyzes input text and divides it into logical segments
     - Ensures each segment contains complete task information
     - Manages parallel processing workflow

   - **Worker Agents**: 
     - Process segments in parallel
     - Extract task details (title, project, time, actions)
     - Map tasks to correct Notion projects
     - Handle time calculations and project matching

   - **Aggregator Agent**:
     - Combines results from all workers
     - Validates and formats task data
     - Prepares final output for Notion sync

3. **Notion Integration Stage** (via MCP):
   - Uses MCP Notion Server tools for reliable API communication
   - Handles authentication and rate limiting
   - Provides structured API calls for:
     - Project synchronization
     - Task creation and updates
     - Time entry management
     - Comment and discussion threading
   - Ensures data consistency between local and Notion states

The MCP (Multi-Channel Protocol) integration is built on top of the [mcp-notion-server](https://github.com/suekou/mcp-notion-server) toolkit, providing robust and type-safe interactions with Notion's API.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
brew install flac  # For macOS (or apt-get install flac for Linux)
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
# Run the application (includes both UI and backend)
python src/main.py
```

Access the application at http://localhost:8000

## Project Structure

```
speech_to_task/
├── src/
│   ├── main.py              # Main application entry point
│   ├── task_agent_extractor.py # LangGraph agent orchestration
│   ├── notion_integration.py   # Notion API integration
│   └── models/              # Data models and state management
├── audio_input/            # Temporary audio storage
├── transcripts/           # Saved transcriptions
├── logs/                  # Application logs
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Next Steps

Planned improvements:

1. Enhanced Task Context
   - Better handling of project relationships
   - Improved time span detection
   - Task priority inference

2. Advanced Audio Processing
   - Real-time transcription
   - Background noise reduction
   - Speaker diarization

3. UI Improvements
   - Dark mode support
   - Keyboard shortcuts
   - Task editing interface

4. Integration Enhancements
   - Additional project management tools
   - Calendar integration
   - Custom workflow automation

## Troubleshooting

### Audio Processing Issues
- **Audio not recognized**: 
  - Check your microphone is working correctly
  - Ensure you have flac installed: `brew install flac` on macOS or `apt-get install flac` on Linux
  - Verify the audio is saved correctly by checking the audio_input directory

### OpenRouter API Issues
- **Task extraction not working**:
  - Verify your OpenRouter API key is valid and correctly set
  - Check the OpenRouter dashboard for quota or rate limiting issues
  - Look at application logs for detailed error messages

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OPENROUTER_API_KEY | Your OpenRouter API key | Required |
| NOTION_API_KEY | Your Notion API key | Optional |
| NOTION_DATABASE_ID | Your Notion database ID | Optional |
| DEBUG | Enable debug mode | 0 (disabled) |

## Requirements

- Python 3.8+
- OpenRouter API key
- Notion API key and database ID (optional)
- FLAC audio codec

## License

MIT
