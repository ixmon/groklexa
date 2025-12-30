# XAI Voice API Access Note

## Current Status

The HTTP endpoint `/audio/transcriptions` exists but returns:
```
403 Forbidden: Team is not authorized to perform this action
```

This indicates:
- ✅ The endpoint exists and is correct
- ✅ Your API key is valid
- ❌ Your team/API key doesn't have permission to use the Voice API endpoints

## Solutions

### Option 1: Request Voice API Access
Contact XAI support to request access to Voice API features for your team/API key.

### Option 2: Use WebSocket API (Recommended)
The XAI Voice API is primarily designed for WebSocket connections:

**Endpoint:** `wss://api.x.ai/v1/realtime`

This is the official way to use the Voice API according to XAI documentation:
- https://docs.x.ai/docs/guides/voice
- https://docs.x.ai/docs/guides/voice/agent

### Option 3: Use LiveKit Integration
XAI provides official LiveKit integration for voice interactions:
- Package: `livekit-agents[xai]`
- Documentation: https://docs.livekit.io/agents/models/realtime/plugins/xai

## Current Implementation

The current HTTP wrapper will work once you have Voice API permissions, but for now you may need to:

1. **Request access** from XAI for Voice API features
2. **Switch to WebSocket** implementation for real-time voice
3. **Use LiveKit** for production voice applications

## Testing

To test if your API key has Voice API access:
```bash
uv run python test_endpoints.py your-audio.mp3
```

If you get 403 errors, you'll need to request access or use WebSocket.

