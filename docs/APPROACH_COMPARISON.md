# HTTP-Based vs LiveKit/WebSocket: Key Differences

## Quick Summary

| Aspect | HTTP-Based | LiveKit/WebSocket |
|--------|-----------|-------------------|
| **Connection** | New connection per request | Persistent connection |
| **Latency** | Higher (100-500ms+) | Lower (50-200ms) |
| **Complexity** | Simple | More complex |
| **Use Case** | One-off requests | Real-time conversations |
| **Setup** | Just HTTP library | LiveKit server + agents |
| **Cost** | Pay per request | Pay per connection time |

## Detailed Comparison

### 1. **Connection Model**

**HTTP-Based:**
```
Client → [Open Connection] → Send Request → Wait → Receive Response → [Close Connection]
```
- Each request opens a new connection
- Stateless - no connection maintained between requests
- Like making a phone call, hanging up, then calling again

**LiveKit/WebSocket:**
```
Client → [Open Persistent Connection] → Stream Data ↔ Stream Data → [Keep Open]
```
- One connection stays open for the entire session
- Stateful - connection remembers context
- Like a phone call that stays connected

### 2. **Latency & Performance**

**HTTP-Based:**
- **Latency**: 200-1000ms+ per request
  - Connection overhead: ~50-200ms
  - Request processing: ~100-500ms
  - Response delivery: ~50-200ms
- **Best for**: Non-real-time use cases
  - Converting a pre-recorded audio file
  - Batch processing multiple files
  - When you can wait a second or two

**LiveKit/WebSocket:**
- **Latency**: 50-300ms for streaming
  - No connection overhead after initial setup
  - Streaming allows immediate processing
  - Bidirectional data flow
- **Best for**: Real-time conversations
  - Voice assistants
  - Live customer support
  - Interactive voice applications

### 3. **Code Complexity**

**HTTP-Based:**
```python
# Simple - just send a request
import requests

response = requests.post(
    "https://api.x.ai/v1/voice/infer",
    headers={"Authorization": f"Bearer {api_key}"},
    files={"audio": audio_data}
)
result = response.json()
```
- ~10-20 lines of code
- Easy to understand and debug
- Standard HTTP libraries (requests, httpx)

**LiveKit/WebSocket:**
```python
# More complex - manage connection lifecycle
from livekit.agents import AgentSession
from livekit.plugins import xai

session = AgentSession(
    llm=xai.realtime.RealtimeModel(voice="Ara")
)

async with session as s:
    # Handle audio streams, events, callbacks
    # Manage connection state
    # Handle reconnection logic
    ...
```
- ~50-200+ lines of code
- Requires understanding of async/await
- Need to handle connection lifecycle
- More dependencies (livekit-agents, websocket libraries)

### 4. **Use Cases**

**HTTP-Based is better for:**
- ✅ Converting a single audio file to text
- ✅ One-off voice synthesis requests
- ✅ Batch processing (convert 100 files)
- ✅ Simple integrations
- ✅ When you don't need real-time responses
- ✅ Learning/prototyping

**LiveKit/WebSocket is better for:**
- ✅ Real-time voice conversations
- ✅ Voice assistants that respond immediately
- ✅ Interactive applications
- ✅ Low-latency requirements (<200ms)
- ✅ Continuous dialogue
- ✅ Production voice applications

### 5. **Setup Requirements**

**HTTP-Based:**
```bash
pip install requests
# That's it!
```
- Just need HTTP library
- No additional services
- Works anywhere

**LiveKit/WebSocket:**
```bash
pip install "livekit-agents[xai]"
# May also need:
# - LiveKit server (or cloud service)
# - WebSocket support
# - Async runtime setup
```
- More dependencies
- May need LiveKit server
- More configuration

### 6. **Error Handling**

**HTTP-Based:**
- Simple: Check status code, parse error JSON
- Each request is independent
- Easy to retry failed requests

**LiveKit/WebSocket:**
- More complex: Handle connection drops, reconnection
- Need to manage connection state
- Stream errors require special handling

### 7. **Cost Implications**

**HTTP-Based:**
- Pay per API call
- Predictable costs
- Good for occasional use

**LiveKit/WebSocket:**
- Pay per connection time
- May be more expensive for short interactions
- Better value for long conversations

## Recommendation for Your Use Case

Since you want a **"basic inference wrapper"**, I'd recommend:

### Start with HTTP-Based if:
- You want something simple and quick
- You're prototyping or learning
- You don't need real-time responses
- You want easy debugging

### Use LiveKit/WebSocket if:
- You need real-time voice conversations
- Latency is critical (<200ms)
- You're building a production voice assistant
- You need bidirectional streaming

## Hybrid Approach

We could also build a wrapper that supports **both**:
- Simple HTTP method for basic inference
- Optional LiveKit method for real-time use
- Same interface, different backends

This gives you flexibility to choose based on your needs!

