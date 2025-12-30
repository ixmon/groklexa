# WebSocket API Usage Guide

## Overview

The WebSocket API (`wss://api.x.ai/v1/realtime`) is the recommended way to use the XAI Voice API. Unlike HTTP endpoints, it doesn't require special permissions and works with standard API keys.

## Quick Start

```python
import asyncio
from groklexa import XAIVoiceWebSocketWrapper

async def main():
    # Simple usage - connects, sends, disconnects automatically
    wrapper = XAIVoiceWebSocketWrapper(voice="Ara")
    result = await wrapper.infer("audio.mp3")
    print(result)

asyncio.run(main())
```

## Connection Management

### Option 1: Automatic (Recommended for simple use)

```python
wrapper = XAIVoiceWebSocketWrapper(voice="Ara")
result = await wrapper.infer("audio.mp3")  # Handles connect/disconnect
```

### Option 2: Context Manager

```python
async with XAIVoiceWebSocketWrapper(voice="Ara") as ws:
    result = await ws.send_audio("audio.mp3")
    # Connection automatically closed when exiting context
```

### Option 3: Manual Management

```python
wrapper = XAIVoiceWebSocketWrapper(voice="Ara")
await wrapper.connect()
try:
    result = await wrapper.send_audio("audio.mp3")
finally:
    await wrapper.disconnect()
```

## Streaming Responses

You can handle streaming responses with a callback:

```python
def on_response(response):
    print(f"Received: {response}")

async with XAIVoiceWebSocketWrapper(voice="Ara") as ws:
    result = await ws.send_audio("audio.mp3", on_response=on_response)
```

## Message Format

**Note:** The exact message format may need adjustment based on the XAI realtime API specification. The current implementation uses:

```json
{
    "type": "input_audio_buffer.append",
    "audio": "<base64_encoded_audio>",
    "voice": "Ara",
    "format": "mp3"
}
```

If you encounter errors, you may need to adjust the message format. Check the [XAI Voice API documentation](https://docs.x.ai/docs/guides/voice) for the exact protocol.

## Error Handling

```python
try:
    wrapper = XAIVoiceWebSocketWrapper(voice="Ara")
    result = await wrapper.infer("audio.mp3")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except TimeoutError as e:
    print(f"Request timed out: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Available Voices

- Ara
- Rex
- Sal
- Eve
- Leo

## Testing

Run the example script:

```bash
uv run python example_websocket.py
```

## Troubleshooting

1. **Connection fails**: Check your API key and network connection
2. **No response**: The message format may need adjustment - check XAI docs
3. **Timeout errors**: Increase timeout or check API status

## References

- [XAI Voice API Documentation](https://docs.x.ai/docs/guides/voice)
- [XAI Realtime API Guide](https://docs.x.ai/docs/guides/voice/agent)

