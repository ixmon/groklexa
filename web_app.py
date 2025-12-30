"""
Web application for XAI Voice API with microphone recording.
"""

import os
import ssl
import tempfile
import asyncio
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from groklexa import XAIVoiceWebSocketWrapper

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


@app.route('/static/openwakeword/<path:filename>')
def serve_openwakeword(filename):
    """Serve OpenWakeWord static files (models, sounds)."""
    return send_from_directory('static/openwakeword', filename)


@app.route('/static/openwakeword/models/<path:filename>')
def serve_openwakeword_models(filename):
    """Serve OpenWakeWord ONNX models."""
    return send_from_directory('static/openwakeword/models', filename)


@app.route('/static/videos/<path:filename>')
def serve_videos(filename):
    """Serve video files."""
    return send_from_directory('static/videos', filename)


# ========== API CONFIGURATION ==========

CONFIG_DIR = Path(__file__).parent / 'config'
CONFIG_FILE = CONFIG_DIR / 'api_settings.json'
CONFIG_EXAMPLE = CONFIG_DIR / 'api_settings.example.json'

# Placeholder for unchanged auth strings
AUTH_UNCHANGED = '__UNCHANGED__'
AUTH_MASK = '••••••••••••'


def load_config():
    """Load API configuration from file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    elif CONFIG_EXAMPLE.exists():
        with open(CONFIG_EXAMPLE, 'r') as f:
            return json.load(f)
    else:
        # Default config
        return {
            "mode": "single",
            "single": {
                "provider": "xai_realtime",
                "url": "wss://api.x.ai/v1/realtime",
                "auth": os.getenv('XAI_API_KEY', ''),
                "protocol": "xai_realtime"
            },
            "transcription": {
                "provider": "browser",
                "url": "",
                "auth": "",
                "protocol": "browser_speech_api"
            },
            "inference": {
                "provider": "grok",
                "url": "https://api.x.ai/v1/chat/completions",
                "auth": os.getenv('XAI_API_KEY', ''),
                "protocol": "openai_compatible",
                "model": "grok-3"
            },
            "synthesis": {
                "provider": "grok",
                "url": "wss://api.x.ai/v1/realtime",
                "auth": os.getenv('XAI_API_KEY', ''),
                "protocol": "xai_realtime",
                "voice": "Ara"
            }
        }


def save_config(config):
    """Save API configuration to file."""
    CONFIG_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def mask_auth(config):
    """Mask auth strings for safe transmission to frontend."""
    masked = json.loads(json.dumps(config))  # Deep copy
    
    for section in ['single', 'transcription', 'inference', 'synthesis']:
        if section in masked and 'auth' in masked[section]:
            if masked[section]['auth']:
                masked[section]['auth'] = AUTH_MASK
    
    return masked


def detect_model_from_url(url):
    """Try to detect model name from URL."""
    if not url:
        return None
    
    url_lower = url.lower()
    
    # Common model patterns
    model_patterns = [
        ('grok-4', 'Grok 4'),
        ('grok-3', 'Grok 3'),
        ('grok-2', 'Grok 2'),
        ('grok-beta', 'Grok Beta'),
        ('claude-3-opus', 'Claude 3 Opus'),
        ('claude-3-sonnet', 'Claude 3 Sonnet'),
        ('claude-sonnet-4', 'Claude Sonnet 4'),
        ('claude-3-haiku', 'Claude 3 Haiku'),
        ('gpt-4', 'GPT-4'),
        ('gpt-3.5', 'GPT-3.5'),
        ('whisper', 'Whisper'),
        ('llama', 'LLaMA'),
    ]
    
    for pattern, name in model_patterns:
        if pattern in url_lower:
            return name
    
    return None


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get API configuration (with masked auth strings)."""
    config = load_config()
    masked = mask_auth(config)
    
    # Add detected models
    for section in ['single', 'transcription', 'inference', 'synthesis']:
        if section in config and 'url' in config[section]:
            detected = detect_model_from_url(config[section]['url'])
            if detected:
                masked[section]['detected_model'] = detected
    
    return jsonify({
        'success': True,
        'config': masked
    })


@app.route('/api/config', methods=['POST'])
def set_config():
    """Save API configuration."""
    try:
        new_config = request.json
        if not new_config:
            return jsonify({'success': False, 'error': 'No configuration provided'}), 400
        
        # Load existing config to preserve unchanged auth strings
        existing_config = load_config()
        
        # Process each section
        for section in ['single', 'transcription', 'inference', 'synthesis']:
            if section in new_config and 'auth' in new_config[section]:
                # If auth is masked placeholder, keep existing
                if new_config[section]['auth'] in [AUTH_UNCHANGED, AUTH_MASK, '']:
                    if section in existing_config and 'auth' in existing_config[section]:
                        new_config[section]['auth'] = existing_config[section]['auth']
        
        save_config(new_config)
        
        return jsonify({
            'success': True,
            'message': 'Configuration saved'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/config/test', methods=['POST'])
def test_connection():
    """Test connection to an API endpoint."""
    try:
        data = request.json
        provider = data.get('provider', '')
        url = data.get('url', '')
        auth = data.get('auth', '')
        protocol = data.get('protocol', '')
        
        # If auth is masked, get from saved config
        if auth in [AUTH_UNCHANGED, AUTH_MASK]:
            config = load_config()
            section = data.get('section', 'single')
            if section in config and 'auth' in config[section]:
                auth = config[section]['auth']
        
        # Test based on protocol
        if protocol == 'xai_realtime':
            # Test WebSocket connection
            async def test_ws():
                import websockets
                headers = {"Authorization": f"Bearer {auth}"}
                try:
                    async with websockets.connect(url, additional_headers=headers) as ws:
                        # Wait for initial message
                        msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        return True, "Connected successfully"
                except asyncio.TimeoutError:
                    return False, "Connection timeout"
                except Exception as e:
                    return False, str(e)
            
            success, message = asyncio.run(test_ws())
            
        elif protocol in ['openai_compatible', 'anthropic_messages']:
            # Test REST endpoint with a simple request
            import requests
            headers = {"Authorization": f"Bearer {auth}"}
            if protocol == 'anthropic_messages':
                headers = {
                    "x-api-key": auth,
                    "anthropic-version": "2023-06-01"
                }
            
            try:
                # Just check if we can reach the endpoint
                resp = requests.get(url.replace('/chat/completions', '/models'), 
                                   headers=headers, timeout=10)
                if resp.status_code in [200, 401, 403]:
                    # 401/403 means we reached the API but auth failed
                    if resp.status_code == 200:
                        success, message = True, "Connected successfully"
                    else:
                        success, message = False, f"Auth error: {resp.status_code}"
                else:
                    success, message = False, f"HTTP {resp.status_code}"
            except Exception as e:
                success, message = False, str(e)
        
        elif protocol == 'browser_speech_api':
            # Browser API doesn't need server-side testing
            success, message = True, "Browser Speech API (tested client-side)"
        
        else:
            success, message = False, f"Unknown protocol: {protocol}"
        
        return jsonify({
            'success': success,
            'message': message,
            'detected_model': detect_model_from_url(url)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


# Available voices for XAI API
AVAILABLE_VOICES = ["Ara", "Rex", "Sal", "Eve", "Leo"]


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/voices')
def get_voices():
    """Get list of available voices."""
    return jsonify({
        'voices': AVAILABLE_VOICES
    })


@app.route('/api/infer/websocket', methods=['POST'])
def infer_websocket():
    """Handle audio inference request using WebSocket (recommended)."""
    try:
        # Get voice parameter
        voice = request.form.get('voice', 'Ara')
        
        # Get conversation history if provided
        conversation_history = []
        if 'conversation_history' in request.form:
            try:
                conversation_history = json.loads(request.form['conversation_history'])
                print(f"Received conversation history: {len(conversation_history)} messages")
            except json.JSONDecodeError:
                print("Failed to parse conversation history")
        
        # Check if audio file is in request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Perform inference using WebSocket
            async def run_inference():
                print(f"Starting WebSocket inference with voice: {voice}")
                wrapper = XAIVoiceWebSocketWrapper(voice=voice)
                result = await wrapper.infer(tmp_path, conversation_history=conversation_history)
                print(f"WebSocket inference complete: {result}")
                return result
            
            # Run async function in event loop
            result = asyncio.run(run_inference())
            
            return jsonify({
                'success': True,
                'result': result
            })
        except Exception as e:
            print(f"WebSocket inference error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    except Exception as e:
        print(f"Request error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Check for API key
    if not os.getenv('XAI_API_KEY'):
        print("Warning: XAI_API_KEY not set. The app will not work without it.")
    
    # SSL certificate paths
    cert_dir = Path(__file__).parent / 'certs'
    cert_file = cert_dir / 'cert.pem'
    key_file = cert_dir / 'key.pem'
    
    # Check if certificates exist
    use_https = cert_file.exists() and key_file.exists()
    
    ssl_context = None
    if use_https:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(cert_file, key_file)
        print("Starting HTTPS web server...")
        print("Open https://spark.home.arpa:5001 in your browser")
        print("Note: Your browser will show a security warning for the self-signed certificate.")
        print("      Click 'Advanced' and 'Proceed' to accept it.")
    else:
        print("Starting HTTP web server (no SSL certificate found)...")
        print("Run './generate_cert.sh' to generate a certificate for HTTPS support.")
        print("Open http://localhost:5001 in your browser")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5001,
        ssl_context=ssl_context
    )

