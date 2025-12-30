"""
Web application for XAI Voice API with microphone recording.
"""

import os
import ssl
import tempfile
import asyncio
import json
import logging
import requests
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from groklexa import XAIVoiceWebSocketWrapper

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('groklexa')

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
                logger.info(f"Received conversation history: {len(conversation_history)} messages")
            except json.JSONDecodeError:
                logger.warning("Failed to parse conversation history")
        
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
                logger.info(f"Starting WebSocket inference with voice: {voice}")
                wrapper = XAIVoiceWebSocketWrapper(voice=voice)
                result = await wrapper.infer(tmp_path, conversation_history=conversation_history)
                logger.info(f"WebSocket inference complete")
                logger.debug(f"Result keys: {result.keys() if result else 'None'}")
                return result
            
            # Run async function in event loop
            result = asyncio.run(run_inference())
            
            return jsonify({
                'success': True,
                'result': result
            })
        except Exception as e:
            logger.error(f"WebSocket inference error: {e}", exc_info=True)
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
        logger.error(f"Request error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ========== SEPARATE API ENDPOINTS ==========

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """Transcribe audio to text using configured transcription API."""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        logger.info(f"Transcription request: {audio_file.filename}")
        
        # Load config
        config = load_config()
        mode = config.get('mode', 'single')
        
        if mode == 'single':
            api_config = config.get('single', {})
        else:
            api_config = config.get('transcription', {})
        
        provider = api_config.get('provider', 'browser')
        
        if provider == 'browser':
            # Browser-based transcription should be handled client-side
            return jsonify({
                'success': True,
                'use_browser': True,
                'message': 'Use browser Web Speech API for transcription'
            })
        
        url = api_config.get('url', '')
        auth = api_config.get('auth', '')
        
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            if provider in ['whisper', 'openai_whisper']:
                transcription = transcribe_with_whisper(url or 'https://api.openai.com/v1/audio/transcriptions', 
                                                         auth, tmp_path)
            elif provider == 'grok':
                # Grok uses the same Whisper-compatible endpoint
                transcription = transcribe_with_whisper(url or 'https://api.x.ai/v1/audio/transcriptions',
                                                         auth, tmp_path)
            elif provider == 'google':
                transcription = transcribe_with_google(url, auth, tmp_path)
            else:
                # Custom provider - assume Whisper-compatible
                transcription = transcribe_with_whisper(url, auth, tmp_path)
            
            logger.info(f"Transcription result: {transcription[:100]}...")
            
            return jsonify({
                'success': True,
                'transcription': transcription
            })
            
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def transcribe_with_whisper(url: str, auth: str, audio_path: str) -> str:
    """Transcribe audio using OpenAI Whisper-compatible API."""
    logger.debug(f"Calling Whisper API: {url}")
    
    headers = {
        'Authorization': f'Bearer {auth}'
    }
    
    with open(audio_path, 'rb') as f:
        files = {
            'file': (os.path.basename(audio_path), f, 'audio/webm'),
            'model': (None, 'whisper-1')
        }
        response = requests.post(url, headers=headers, files=files, timeout=60)
    
    response.raise_for_status()
    data = response.json()
    return data.get('text', '')


def transcribe_with_google(url: str, auth: str, audio_path: str) -> str:
    """Transcribe audio using Google Cloud Speech-to-Text API."""
    logger.debug(f"Calling Google Speech API: {url}")
    
    import base64
    
    headers = {
        'Authorization': f'Bearer {auth}',
        'Content-Type': 'application/json'
    }
    
    with open(audio_path, 'rb') as f:
        audio_content = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        'config': {
            'encoding': 'WEBM_OPUS',
            'sampleRateHertz': 48000,
            'languageCode': 'en-US'
        },
        'audio': {
            'content': audio_content
        }
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    results = data.get('results', [])
    if results:
        return results[0].get('alternatives', [{}])[0].get('transcript', '')
    return ''


@app.route('/api/infer/text', methods=['POST'])
def infer_text():
    """Text-based inference - takes transcribed text, returns AI response."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        text = data.get('text', '')
        conversation_history = data.get('conversation_history', [])
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        logger.info(f"Text inference request: {text[:100]}...")
        
        # Load config to determine which provider to use
        config = load_config()
        mode = config.get('mode', 'single')
        
        if mode == 'single':
            # Use single API config
            api_config = config.get('single', {})
        else:
            # Use separate inference config
            api_config = config.get('inference', {})
        
        provider = api_config.get('provider', 'grok')
        url = api_config.get('url', '')
        auth = api_config.get('auth', '')
        protocol = api_config.get('protocol', 'openai_compatible')
        model = api_config.get('model', 'grok-3')
        
        logger.debug(f"Using inference provider: {provider}, protocol: {protocol}")
        
        # Build messages array
        messages = []
        for msg in conversation_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            messages.append({'role': role, 'content': content})
        
        # Add current message
        messages.append({'role': 'user', 'content': text})
        
        # Call the appropriate API
        if protocol == 'openai_compatible':
            response_text = call_openai_compatible(url, auth, model, messages)
        elif protocol == 'anthropic_messages':
            response_text = call_anthropic(url, auth, model, messages)
        elif protocol == 'browser_speech_api':
            # This shouldn't be called server-side
            return jsonify({'error': 'Browser Speech API should be used client-side'}), 400
        else:
            return jsonify({'error': f'Unknown protocol: {protocol}'}), 400
        
        logger.info(f"Inference response: {response_text[:100]}...")
        
        return jsonify({
            'success': True,
            'response': response_text
        })
        
    except Exception as e:
        logger.error(f"Text inference error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def call_openai_compatible(url: str, auth: str, model: str, messages: list) -> str:
    """Call an OpenAI-compatible API (Grok, OpenAI, Ollama, etc.)."""
    logger.debug(f"Calling OpenAI-compatible API: {url}, model: {model}")
    
    headers = {
        'Authorization': f'Bearer {auth}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': model,
        'messages': messages
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    return data['choices'][0]['message']['content']


def call_anthropic(url: str, auth: str, model: str, messages: list) -> str:
    """Call Anthropic's Messages API."""
    logger.debug(f"Calling Anthropic API: {url}, model: {model}")
    
    headers = {
        'x-api-key': auth,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json'
    }
    
    # Convert to Anthropic format (system message separate)
    system_msg = None
    anthropic_messages = []
    
    for msg in messages:
        if msg['role'] == 'system':
            system_msg = msg['content']
        else:
            anthropic_messages.append({
                'role': msg['role'],
                'content': msg['content']
            })
    
    payload = {
        'model': model,
        'max_tokens': 4096,
        'messages': anthropic_messages
    }
    
    if system_msg:
        payload['system'] = system_msg
    
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    return data['content'][0]['text']


@app.route('/api/synthesize', methods=['POST'])
def synthesize():
    """Text-to-speech synthesis - takes text, returns audio."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        logger.info(f"Synthesis request: {text[:100]}...")
        
        # Load config
        config = load_config()
        mode = config.get('mode', 'single')
        
        if mode == 'single':
            api_config = config.get('single', {})
        else:
            api_config = config.get('synthesis', {})
        
        provider = api_config.get('provider', 'grok')
        protocol = api_config.get('protocol', 'xai_realtime')
        voice = api_config.get('voice', 'Ara')
        
        if protocol == 'browser_speech_api':
            # Return text for client-side synthesis
            return jsonify({
                'success': True,
                'use_browser': True,
                'text': text
            })
        
        # For XAI realtime, we need to use WebSocket to get audio
        # This is a simplified version - the full audio generation
        # would require the WebSocket wrapper
        logger.warning("Server-side synthesis for separate APIs not yet implemented")
        return jsonify({
            'success': True,
            'use_browser': True,  # Fallback to browser
            'text': text
        })
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}", exc_info=True)
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

