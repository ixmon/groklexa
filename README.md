# Groklexa

A voice assistant with wake word detection, powered by XAI Realtime API.

## Features

- 🎤 **Wake word detection** - Say "Groklexa" to wake, say it again to sleep
- 🔊 **Voice conversations** - Real-time speech-to-text, inference, and text-to-speech
- 🌙 **Day/Night modes** - Elegant UI with video animations
- 🎯 **Silero VAD** - Neural network-based voice activity detection
- ⚙️ **Configurable APIs** - Use XAI, Anthropic, OpenAI, or local endpoints
- 📋 **Conversation history** - Persistent chat with copy/download options

## Quick Start

### Prerequisites

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) for dependency management
- XAI API key (for default configuration)

### Installation

```bash
# Clone and install dependencies
git clone https://github.com/yourusername/groklexa.git
cd groklexa
uv sync
```

### Setup

Set your XAI API key:

```bash
export XAI_API_KEY="your-api-key-here"
```

### Run

```bash
uv run python web_app.py
```

Open https://localhost:5001 in your browser.

## Usage

1. Click "Listen" toggle to enable wake word detection
2. Say "Groklexa" to activate conversation mode
3. Speak naturally - VAD detects when you're talking
4. Say "Groklexa" again to return to sleep mode

## HTTPS for Local Network

To access from other devices on your local network (required for microphone):

1. **Generate SSL certificate:**
   ```bash
   mkdir -p certs
   openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes \
     -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,DNS:*.home.arpa,IP:127.0.0.1"
   ```

2. **Start server and access via HTTPS**

3. **Accept the self-signed certificate in your browser**

## Configuration

Open the settings panel (gear icon) to configure:

- **Voice selection** - Choose from Ara, Rex, Sal, Eve, Leo
- **API configuration** - Single unified API or separate APIs for transcription, inference, and synthesis
- **Manual VAD** - Toggle volume-based VAD for testing

Configuration is saved to `config/api_settings.json` (gitignored).

## Project Structure

```
groklexa/
├── groklexa/
│   ├── __init__.py
│   └── xai_voice_websocket.py    # WebSocket wrapper for XAI API
├── templates/
│   └── index.html                 # Web interface
├── static/
│   ├── openwakeword/             # Wake word models
│   │   └── models/               # ONNX models (melspec, embedding, VAD, groklexa)
│   └── videos/                   # UI animations
├── config/
│   └── api_settings.json         # API configuration (gitignored)
├── docs/                         # Documentation
├── web_app.py                    # Flask web server
├── pyproject.toml                # Project configuration
└── README.md
```

## Python API

```python
import asyncio
from groklexa import XAIVoiceWebSocketWrapper

async def main():
    async with XAIVoiceWebSocketWrapper(voice="Ara") as ws:
        result = await ws.send_audio("path/to/audio.wav")
        print(result['transcription'])
        print(result['ai_response_text'])
        # result['ai_response_audio'] contains PCM16 audio bytes

asyncio.run(main())
```

## Technology

- **XAI Realtime API** - Unified transcription, inference, and synthesis via WebSocket
- **OpenWakeWord** - Browser-based wake word detection with ONNX Runtime
- **Silero VAD** - Neural network voice activity detection
- **Flask** - Python web server
- **Web Audio API** - Real-time audio processing

## Future Plans

- [ ] Command-line interface
- [ ] Mobile app (iOS/Android)
- [ ] Custom wake word training
- [ ] Multi-provider cascading (fallback to local when offline)

## License

MIT
