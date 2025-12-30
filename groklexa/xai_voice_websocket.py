"""
XAI Voice API WebSocket Wrapper

WebSocket-based wrapper for the XAI Voice API using wss://api.x.ai/v1/realtime
"""

import os
import json
import asyncio
import base64
import io
from typing import Union, Optional, Dict, Any, Callable, List
from pathlib import Path
import websockets
from websockets.client import WebSocketClientProtocol

try:
    from pydub import AudioSegment
    from pydub.effects import normalize
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available. Audio normalization disabled.")


def extract_transcription_and_response(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract transcription and AI response from WebSocket messages.
    
    Args:
        responses: List of response messages from the API
    
    Returns:
        Dictionary with transcription, AI text response, and audio response
    """
    transcription = None
    ai_text = ""
    ai_audio_chunks = []
    
    for resp in responses:
        event_type = resp.get("type", "")
        
        # Extract user's transcription
        if "transcript" in resp:
            transcription = resp.get("transcript")
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcription = resp.get("transcript", transcription)
        
        # Extract AI's text response
        if event_type == "response.text.delta":
            ai_text += resp.get("delta", "")
        elif event_type == "response.text.done":
            ai_text = resp.get("text", ai_text)
        
        # Extract AI's audio response
        if event_type == "response.audio.delta":
            delta = resp.get("delta", "")
            if delta:
                ai_audio_chunks.append(delta)
    
    # Combine audio chunks
    ai_audio = None
    if ai_audio_chunks:
        # Audio is base64 encoded, combine and keep as base64
        ai_audio = "".join(ai_audio_chunks)
    
    return {
        "user_speech": transcription,
        "ai_text_response": ai_text if ai_text else None,
        "ai_audio_response": ai_audio
    }


class XAIVoiceWebSocketWrapper:
    """
    WebSocket-based wrapper for the XAI Voice API.
    
    Supports real-time voice interactions via WebSocket connection.
    """
    
    # Available voices for the XAI Voice API
    AVAILABLE_VOICES = ["Ara", "Rex", "Sal", "Eve", "Leo"]
    
    # WebSocket endpoint
    WS_URL = "wss://api.x.ai/v1/realtime"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "Ara",
        ws_url: Optional[str] = None
    ):
        """
        Initialize the XAI Voice API WebSocket wrapper.
        
        Args:
            api_key: XAI API key. If not provided, reads from XAI_API_KEY env var.
            voice: Voice to use for responses. Options: Ara, Rex, Sal, Eve, Leo
            ws_url: WebSocket URL. Defaults to wss://api.x.ai/v1/realtime
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "XAI_API_KEY not found. Provide it as an argument or set the "
                "XAI_API_KEY environment variable."
            )
        
        if voice not in self.AVAILABLE_VOICES:
            raise ValueError(
                f"Invalid voice '{voice}'. Available voices: {', '.join(self.AVAILABLE_VOICES)}"
            )
        self.voice = voice
        
        self.ws_url = ws_url or self.WS_URL
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.connected = False
    
    async def connect(self) -> None:
        """
        Establish WebSocket connection to XAI Voice API.
        
        Raises:
            ConnectionError: If connection fails
        """
        try:
            # Add API key to headers (as per XAI documentation)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "groklexa/0.1.0"
            }
            
            # Use additional_headers for websockets 14.0+ (extra_headers was renamed)
            self.websocket = await websockets.connect(
                self.ws_url,
                additional_headers=headers
            )
            self.connected = True
            
            # Wait for initial connection message (if any)
            try:
                initial_msg = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                if initial_msg:
                    print(f"Connected. Initial message: {initial_msg}")
            except asyncio.TimeoutError:
                pass  # No initial message, that's okay
            
            # Configure session to enable audio output
            # Note: Transcription seems to be automatic, don't need to configure it
            session_update = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "voice": self.voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": None  # Disable automatic turn detection
                }
            }
            print(f"Configuring session for audio output with voice: {self.voice}")
            await self.websocket.send(json.dumps(session_update))
            
            # Wait briefly for session to be configured
            await asyncio.sleep(0.5)
            print("Session update sent")
            
        except Exception as e:
            self.connected = False
            raise ConnectionError(f"Failed to connect to WebSocket: {e}")
    
    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self.websocket and self.connected:
            await self.websocket.close()
            self.connected = False
    
    async def configure_context(self, conversation_history: Optional[List[Dict[str, str]]] = None) -> None:
        """
        Configure session with conversation context via instructions.
        """
        if not conversation_history:
            return
            
        # Build context string
        context_parts = []
        for msg in conversation_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            speaker = "User" if role == "user" else "Grok"
            context_parts.append(f"{speaker}: {content}")
        context = "\n".join(context_parts)
        
        # Update session with instructions containing context
        session_update = {
            "type": "session.update",
            "session": {
                "instructions": f"""You are Grok, a helpful and witty AI assistant.
Continue this conversation naturally. Here's the conversation so far:

{context}

Respond to the user's new audio message, building on this context."""
            }
        }
        
        print(f"Updating session with conversation context ({len(conversation_history)} messages)")
        await self.websocket.send(json.dumps(session_update))
        await asyncio.sleep(0.3)
    
    async def send_audio(
        self,
        audio_data: Union[str, Path, bytes],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        on_response: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Send audio data over WebSocket and wait for response.
        
        Args:
            audio_data: Path to audio file (str/Path) or audio data as bytes
            conversation_history: Optional list of previous messages [{"role": "user"|"assistant", "content": "..."}]
            on_response: Optional callback function for streaming responses
        
        Returns:
            Dictionary containing the API response
        
        Raises:
            ConnectionError: If not connected
            ValueError: If audio file not found
        """
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected. Call connect() first.")
        
        # Load audio data
        if isinstance(audio_data, (str, Path)):
            audio_path = Path(audio_data)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
        else:
            audio_bytes = audio_data
        
        # Normalize audio volume if pydub is available
        if PYDUB_AVAILABLE:
            try:
                print("Normalizing audio volume...")
                audio_io = io.BytesIO(audio_bytes)
                
                # Try to detect and load the audio format
                # Browser MediaRecorder typically outputs WebM/Opus or WebM/VP8
                try:
                    # Try loading as-is (auto-detect format)
                    audio_segment = AudioSegment.from_file(audio_io)
                except Exception as e1:
                    print(f"Auto-detect failed: {e1}, trying specific formats...")
                    audio_io.seek(0)
                    # Try common formats
                    for fmt in ['webm', 'ogg', 'mp3', 'wav']:
                        try:
                            audio_io.seek(0)
                            audio_segment = AudioSegment.from_file(audio_io, format=fmt)
                            print(f"Successfully loaded as {fmt}")
                            break
                        except:
                            continue
                    else:
                        raise ValueError("Could not load audio in any supported format")
                
                # Normalize volume (boost to maximum without clipping)
                normalized = normalize(audio_segment)
                
                # Increase gain by additional 10dB if still quiet
                normalized = normalized + 10
                
                # Convert to PCM16 WAV (more compatible format for speech recognition)
                # Set sample rate to 24kHz (good balance for speech)
                normalized = normalized.set_frame_rate(24000).set_channels(1)
                
                # Export as WAV PCM16
                output = io.BytesIO()
                normalized.export(output, format="wav", parameters=["-acodec", "pcm_s16le"])
                audio_bytes = output.getvalue()
                print(f"Audio normalized and converted to WAV PCM16. Size: {len(audio_bytes)} bytes, "
                      f"Duration: {len(normalized)}ms, dBFS: {normalized.dBFS:.1f}, "
                      f"Sample rate: 24000Hz, Channels: 1")
            except Exception as e:
                print(f"Warning: Could not normalize audio: {e}")
                print("Using original audio...")
        else:
            print("Warning: pydub not available, skipping audio normalization")
            print("Install with: uv add pydub")
        
        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Configure conversation context via session.update instructions
        if conversation_history:
            await self.configure_context(conversation_history)
        
        # Step 1: Send audio data
        # The API expects PCM16 audio data, not MP3
        message = {
            "type": "input_audio_buffer.append",
            "audio": audio_base64
        }
        
        print(f"Sending audio data ({len(audio_base64)} bytes base64, {len(audio_bytes)} bytes raw)")
        await self.websocket.send(json.dumps(message))
        
        # Step 2: Commit the audio buffer to trigger processing
        commit_message = {
            "type": "input_audio_buffer.commit"
        }
        
        print(f"Committing audio buffer")
        await self.websocket.send(json.dumps(commit_message))
        
        # Wait for the audio buffer to be committed and processed
        await asyncio.sleep(0.3)
        
        # Step 3: Request a response with voice output
        # Note: XAI API may not support 'instructions' field - use minimal request
        response_request = {
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "voice": self.voice
            }
        }
        
        print(f"Requesting response with voice: {self.voice}")
        await self.websocket.send(json.dumps(response_request))
        
        # Wait for response with longer timeout for AI inference
        responses = []
        timeout_seconds = 30.0  # Longer timeout for AI to generate response
        has_transcription = False
        has_ai_response = False
        
        try:
            # Collect responses (API may send multiple messages)
            response_count = 0
            max_responses = 50  # Increased to handle full conversation
            
            while response_count < max_responses:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=timeout_seconds)
                response_data = json.loads(response)
                responses.append(response_data)
                response_count += 1
                
                msg_type = response_data.get("type", "")
                # Don't print full response if it contains large audio data
                if "delta" in response_data and len(str(response_data.get("delta", ""))) > 100:
                    print(f"Received message type: {msg_type} (with {len(response_data.get('delta', ''))} bytes data)")
                else:
                    print(f"Received message type: {msg_type}")
                
                # Track what we've received
                if "transcript" in response_data or msg_type == "conversation.item.input_audio_transcription.completed":
                    has_transcription = True
                    print(f"  📝 Transcription received")
                
                if msg_type.startswith("response.") and "delta" in msg_type:
                    has_ai_response = True
                    print(f"  🤖 AI response chunk received")
                
                # Call callback if provided
                if on_response:
                    on_response(response_data)
                
                # Check if this is the final response
                # Wait for both transcription AND AI response before breaking
                if msg_type in ("response.done", "error"):
                    print(f"Final message received: {msg_type}")
                    break
                
                # If we have transcription and AI started responding, wait for completion
                if has_transcription and msg_type in ("response.text.done", "response.output_audio.done"):
                    print(f"AI response completed: {msg_type}")
                    # Wait a bit more for any final messages
                    try:
                        final_response = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
                        final_data = json.loads(final_response)
                        responses.append(final_data)
                        print(f"Received final: {final_data.get('type')}")
                        if final_data.get('type') == 'response.done':
                            break
                    except asyncio.TimeoutError:
                        break
                
                # Skip ping messages but don't break
                if msg_type == "ping":
                    continue
                    
        except asyncio.TimeoutError:
            print(f"Timeout after {timeout_seconds}s, received {len(responses)} responses")
            print(f"Has transcription: {has_transcription}, Has AI response: {has_ai_response}")
            if not responses:
                raise TimeoutError(f"No response received from API after {timeout_seconds} seconds")
        
        # Parse and extract meaningful data from responses
        transcription = None
        ai_response_text = ""
        ai_response_audio = None
        ai_audio_transcript = ""
        all_events = []
        
        for resp in responses:
            event_type = resp.get("type", "")
            all_events.append(event_type)
            
            # Extract transcription (user's speech) - only from user-related events, NOT AI response events
            # Important: Don't extract from response.output_audio_transcript.* events as those are AI's words
            if "transcript" in resp and not event_type.startswith("response."):
                transcription = resp.get("transcript")
                print(f"User Transcription: {transcription}")
            
            # Extract from conversation.item.input_audio_transcription.completed
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcription = resp.get("transcript", transcription)
                print(f"User Speech (from transcription event): {transcription}")
            
            # Extract from conversation.item.added
            if event_type == "conversation.item.added":
                item = resp.get("item", {})
                content = item.get("content", [])
                for c in content:
                    if c.get("type") == "input_audio" and "transcript" in c:
                        transcription = c.get("transcript")
                        print(f"User Speech: {transcription}")
            
            # Extract AI text response (streaming)
            if event_type == "response.text.delta":
                delta = resp.get("delta", "")
                ai_response_text += delta
                print(f"AI text delta: {delta}")
            elif event_type == "response.text.done":
                text = resp.get("text", "")
                if text:
                    ai_response_text = text
                print(f"AI Response Text Complete: {ai_response_text}")
            
            # Extract AI audio transcript (what the AI said) - API uses "output_audio_transcript"
            if event_type == "response.output_audio_transcript.delta":
                delta = resp.get("delta", "")
                ai_audio_transcript += delta
                print(f"AI audio transcript delta: {delta}")
            elif event_type == "response.output_audio_transcript.done":
                transcript = resp.get("transcript", "")
                if transcript:
                    ai_audio_transcript = transcript
                print(f"AI Audio Transcript Complete: {ai_audio_transcript}")
            
            # Extract AI audio response (API uses "output_audio" not just "audio")
            if event_type == "response.output_audio.delta":
                if not ai_response_audio:
                    ai_response_audio = b""
                # Audio is base64 encoded
                audio_delta = resp.get("delta", "")
                if audio_delta:
                    decoded = base64.b64decode(audio_delta)
                    ai_response_audio += decoded
                    print(f"  🎵 Received audio chunk: {len(decoded)} bytes (total: {len(ai_response_audio)})")
            elif event_type == "response.output_audio.done":
                audio_size = len(ai_response_audio) if ai_response_audio else 0
                print(f"AI Response Audio Complete: {audio_size} bytes")
                if audio_size == 0:
                    print("  ⚠️  WARNING: No audio data received!")
        
        # Use audio transcript if text response is empty
        final_ai_response = ai_response_text if ai_response_text else ai_audio_transcript
        
        # Debug audio info
        audio_size = len(ai_response_audio) if ai_response_audio else 0
        print(f"\n🔍 Audio Debug:")
        print(f"  - Audio chunks received: {audio_size > 0}")
        print(f"  - Audio size: {audio_size} bytes")
        print(f"  - Events containing 'audio': {[e for e in all_events if 'audio' in e]}")
        print(f"  - Events containing 'response': {[e for e in all_events if 'response' in e]}")
        print(f"  - All events: {all_events}")
        
        # Return structured response
        result = {
            "type": "response.processed",
            "transcription": transcription,
            "ai_response_text": final_ai_response,
            "ai_audio_transcript": ai_audio_transcript if ai_audio_transcript != final_ai_response else None,
            "ai_response_audio_base64": base64.b64encode(ai_response_audio).decode('utf-8') if ai_response_audio and audio_size > 0 else None,
            "ai_response_audio_bytes": audio_size,
            "events_received": all_events,
            "raw_responses": responses
        }
        
        print(f"\n=== Conversation Summary ===")
        print(f"📝 You said: {transcription or 'N/A'}")
        print(f"🤖 AI responded: {final_ai_response or 'N/A'}")
        if ai_audio_transcript and ai_audio_transcript != final_ai_response:
            print(f"🔊 AI audio said: {ai_audio_transcript}")
        print(f"🎵 Audio bytes: {len(ai_response_audio) if ai_response_audio else 0}")
        print(f"📊 Events: {', '.join(all_events)}")
        print("="*50)
        
        return result
    
    async def stream_audio(
        self,
        audio_stream: bytes,
        chunk_size: int = 4096,
        on_response: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Stream audio data in chunks over WebSocket.
        
        Args:
            audio_stream: Audio data as bytes
            chunk_size: Size of chunks to send
            on_response: Optional callback for responses
        
        Returns:
            Final response dictionary
        """
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected. Call connect() first.")
        
        # Send audio in chunks
        for i in range(0, len(audio_stream), chunk_size):
            chunk = audio_stream[i:i + chunk_size]
            chunk_b64 = base64.b64encode(chunk).decode('utf-8')
            
            message = {
                "type": "audio.input.chunk",
                "audio": chunk_b64,
                "chunk_index": i // chunk_size,
                "is_final": (i + chunk_size) >= len(audio_stream)
            }
            
            await self.websocket.send(json.dumps(message))
        
        # Wait for final response
        responses = []
        try:
            while True:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=30.0)
                response_data = json.loads(response)
                responses.append(response_data)
                
                if on_response:
                    on_response(response_data)
                
                if response_data.get("type") in ("response.complete", "response.done", "error"):
                    break
        except asyncio.TimeoutError:
            pass
        
        return responses[-1] if responses else {}
    
    async def infer(
        self,
        audio_file: Union[str, Path, bytes],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform inference on an audio file using WebSocket.
        
        This is a convenience method that connects, sends audio, and disconnects.
        
        Args:
            audio_file: Path to audio file or audio data as bytes
            conversation_history: Optional list of previous messages [{"role": "user"|"assistant", "content": "..."}]
            **kwargs: Additional parameters
        
        Returns:
            Dictionary containing the API response
        """
        await self.connect()
        try:
            result = await self.send_audio(audio_file, conversation_history)
            return result
        finally:
            await self.disconnect()
    
    def get_available_voices(self) -> List[str]:
        """
        Get list of available voices.
        
        Returns:
            List of available voice names
        """
        return self.AVAILABLE_VOICES.copy()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    def __repr__(self) -> str:
        return f"XAIVoiceWebSocketWrapper(voice='{self.voice}', ws_url='{self.ws_url}')"

