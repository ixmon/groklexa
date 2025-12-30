"""
Example usage of the XAI Voice API WebSocket wrapper.
"""

import asyncio
import os
from groklexa import XAIVoiceWebSocketWrapper


async def main():
    """Example usage of the XAI Voice WebSocket wrapper."""
    
    # Initialize the wrapper
    print("Initializing XAI Voice WebSocket wrapper...")
    wrapper = XAIVoiceWebSocketWrapper(voice="Ara")
    print(f"Wrapper initialized: {wrapper}")
    
    # Show available voices
    print(f"\nAvailable voices: {', '.join(wrapper.get_available_voices())}")
    
    # Example 1: Simple inference (connects, sends, disconnects)
    audio_file = "example.mp3"  # Replace with your MP3 file path
    
    if os.path.exists(audio_file):
        print(f"\nPerforming inference on {audio_file}...")
        try:
            result = await wrapper.infer(audio_file)
            print(f"Success! Response: {result}")
        except Exception as e:
            print(f"Error during inference: {e}")
    else:
        print(f"\nAudio file '{audio_file}' not found.")
        print("To test the wrapper:")
        print(f"  1. Place an MP3 file at '{audio_file}'")
        print("  2. Run this script again")
    
    # Example 2: Using context manager for connection management
    print("\n" + "="*50)
    print("Example: Using context manager")
    async with XAIVoiceWebSocketWrapper(voice="Rex") as ws_wrapper:
        print(f"Connected: {ws_wrapper.connected}")
        
        if os.path.exists(audio_file):
            # Define callback for streaming responses
            def on_response(response):
                print(f"Received: {response.get('type', 'unknown')}")
            
            result = await ws_wrapper.send_audio(audio_file, on_response=on_response)
            print(f"Final result: {result}")
    
    # Example 3: Manual connection management
    print("\n" + "="*50)
    print("Example: Manual connection management")
    ws_wrapper = XAIVoiceWebSocketWrapper(voice="Sal")
    try:
        await ws_wrapper.connect()
        print(f"Connected: {ws_wrapper.connected}")
        
        if os.path.exists(audio_file):
            result = await ws_wrapper.send_audio(audio_file)
            print(f"Result: {result}")
    finally:
        await ws_wrapper.disconnect()
        print(f"Disconnected: {not ws_wrapper.connected}")


if __name__ == "__main__":
    asyncio.run(main())

