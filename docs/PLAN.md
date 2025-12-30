# XAI Voice API Inference Wrapper - Implementation Plan

## Overview
Create a basic inference wrapper around the XAI Voice API using the `XAI_API_KEY` from the environment.

## Research Findings
- XAI API key is available in environment: `XAI_API_KEY`
- XAI Voice API supports real-time voice interactions
- Available voices: Ara, Rex, Sal, Eve, Leo
- API can be accessed via:
  1. LiveKit Agents (WebSocket-based streaming)
  2. Direct HTTP REST API (if available)

## Implementation Plan

### 1. Project Structure
```
groklexa/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── xai_voice_wrapper.py      # Main wrapper class
├── example_usage.py          # Example script
└── .env.example              # Example environment file
```

### 2. Core Components

#### A. XAIVoiceWrapper Class
- **Purpose**: Main wrapper class for XAI Voice API interactions
- **Features**:
  - Load API key from environment variable `XAI_API_KEY`
  - Support for different voice options
  - Handle audio input/output
  - Error handling and retry logic
  - Support both streaming and non-streaming modes

#### B. Key Methods
1. `__init__(voice="Ara")` - Initialize with voice selection
2. `infer(audio_input, **kwargs)` - Main inference method
3. `stream_infer(audio_input, **kwargs)` - Streaming inference
4. `get_available_voices()` - List available voices
5. `validate_api_key()` - Check API key validity

### 3. Implementation Approach

**Option A: LiveKit-based (Recommended for real-time)**
- Use `livekit-agents[xai]` package
- WebSocket-based streaming
- Lower latency for real-time interactions
- More complex setup

**Option B: Direct HTTP API (Simpler)**
- Direct REST API calls
- Simpler implementation
- May have higher latency
- Easier to debug

**Decision**: Start with Option B (HTTP-based) for simplicity, but design to allow Option A extension.

### 4. Dependencies
- `requests` - HTTP requests
- `python-dotenv` - Environment variable management (optional)
- `aiohttp` - Async HTTP (if async support needed)

### 5. Error Handling
- API key validation
- Network error handling
- Rate limiting handling
- Invalid input validation
- Response parsing errors

### 6. Example Usage
```python
from xai_voice_wrapper import XAIVoiceWrapper

# Initialize wrapper
wrapper = XAIVoiceWrapper(voice="Ara")

# Perform inference
response = wrapper.infer(audio_data)
print(response)
```

## Next Steps
1. Research exact API endpoint URLs and request formats
2. Implement basic HTTP wrapper
3. Add error handling
4. Create example usage
5. Test with actual API calls

