"""
Web application for XAI Voice API with microphone recording.
"""

import os
import ssl
import tempfile
import asyncio
import json
import logging
import time
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
        # Default config - no env fallback, user must configure via UI
        return {
            "mode": "single",
            "single": {
                "provider": "xai_realtime",
                "url": "wss://api.x.ai/v1/realtime",
                "auth": "",  # User must configure via settings
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
                "auth": "",  # User must configure via settings
                "protocol": "openai_compatible",
                "model": "grok-3"
            },
            "synthesis": {
                "provider": "grok",
                "url": "wss://api.x.ai/v1/realtime",
                "auth": "",  # User must configure via settings
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
            
        elif protocol in ['openai_compatible', 'grok', 'openai', 'ollama', 'anthropic_messages', 'anthropic']:
            # Test REST endpoint with a simple request
            headers = {"Authorization": f"Bearer {auth}"}
            if protocol in ['anthropic_messages', 'anthropic']:
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


# Available voices by provider
PROVIDER_VOICES = {
    'grok': ['Ara', 'Rex', 'Sal', 'Eve', 'Leo'],
    'xai_realtime': ['Ara', 'Rex', 'Sal', 'Eve', 'Leo'],
    'openai_tts': ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
    'browser': [],  # Fetched client-side
}

# Cache for edge-tts voices
_edge_tts_voices_cache = None

# Local Whisper model (lazy loaded)
_whisper_model = None
_whisper_model_size = 'base'  # Options: tiny, base, small, medium, large-v3


def get_whisper_model():
    """Get or initialize the local Whisper model."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    
    try:
        from faster_whisper import WhisperModel
        logger.info(f"Loading Whisper model: {_whisper_model_size}")
        # Use CPU by default, can be changed to 'cuda' for GPU
        _whisper_model = WhisperModel(_whisper_model_size, device="cpu", compute_type="int8")
        logger.info("Whisper model loaded successfully")
        return _whisper_model
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return None


def transcribe_with_local_whisper(audio_path: str) -> str:
    """Transcribe audio using local Whisper model."""
    model = get_whisper_model()
    if model is None:
        raise Exception("Whisper model not available")
    
    logger.debug(f"Transcribing with local Whisper: {audio_path}")
    
    try:
        segments, info = model.transcribe(audio_path, beam_size=5)
        transcript = " ".join([segment.text for segment in segments])
        logger.info(f"Local Whisper transcription: {transcript[:100]}...")
        return transcript.strip()
    except Exception as e:
        logger.error(f"Local Whisper transcription error: {e}")
        raise


async def get_edge_tts_voices():
    """Get available Edge TTS voices (cached)."""
    global _edge_tts_voices_cache
    if _edge_tts_voices_cache is not None:
        return _edge_tts_voices_cache
    
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
        # Filter to English voices and format nicely
        english_voices = [
            {
                'id': v['ShortName'],
                'name': f"{v['ShortName'].replace('Neural', '')} ({v['Locale']})",
                'gender': v.get('Gender', 'Unknown'),
                'locale': v['Locale']
            }
            for v in voices
            if v['Locale'].startswith('en-')
        ]
        _edge_tts_voices_cache = english_voices
        return english_voices
    except Exception as e:
        logger.error(f"Failed to get Edge TTS voices: {e}")
        return []


@app.route('/api/whisper/status')
def whisper_status():
    """Get local Whisper model status."""
    global _whisper_model, _whisper_model_size
    return jsonify({
        'success': True,
        'loaded': _whisper_model is not None,
        'model_size': _whisper_model_size,
        'available_sizes': ['tiny', 'base', 'small', 'medium', 'large-v3']
    })


@app.route('/api/whisper/load', methods=['POST'])
def whisper_load():
    """Load or reload Whisper model with specified size."""
    global _whisper_model, _whisper_model_size
    
    data = request.json or {}
    size = data.get('size', 'base')
    
    if size not in ['tiny', 'base', 'small', 'medium', 'large-v3']:
        return jsonify({'success': False, 'error': f'Invalid model size: {size}'}), 400
    
    # Unload existing model
    _whisper_model = None
    _whisper_model_size = size
    
    # Try to load new model
    model = get_whisper_model()
    if model:
        return jsonify({
            'success': True,
            'message': f'Whisper model {size} loaded successfully'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to load Whisper model'
        }), 500


@app.route('/api/models/<provider>')
def get_provider_models(provider):
    """Get available models for a specific inference provider.
    
    Models are cached and only refreshed if:
    - Cache is older than 7 days
    - Force refresh is requested (?refresh=true)
    - No cached models exist for this provider
    """
    try:
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        # Load config to get auth and cached models
        config = load_config()
        mode = config.get('mode', 'single')
        
        if mode == 'single':
            api_config = config.get('single', {})
        else:
            api_config = config.get('inference', {})
        
        auth = api_config.get('auth', '')
        
        # Check cache
        cached_models = config.get('_cached_models', {})
        provider_cache = cached_models.get(provider, {})
        cached_at = provider_cache.get('cached_at', 0)
        cached_list = provider_cache.get('models', [])
        
        # Cache duration: 7 days in seconds
        CACHE_DURATION = 7 * 24 * 60 * 60
        cache_age = time.time() - cached_at
        cache_valid = cache_age < CACHE_DURATION and len(cached_list) > 0
        
        if cache_valid and not force_refresh:
            logger.info(f"Using cached models for {provider} (age: {cache_age/3600:.1f} hours)")
            return jsonify({'success': True, 'models': cached_list, 'cached': True})
        
        # Fetch fresh models
        models = []
        fetch_success = False
        
        if provider == 'grok':
            models = fetch_openai_compatible_models('https://api.x.ai/v1/models', auth)
            fetch_success = len(models) > 0
        elif provider == 'openai':
            models = fetch_openai_compatible_models('https://api.openai.com/v1/models', auth)
            fetch_success = len(models) > 0
        elif provider == 'ollama':
            models = fetch_ollama_models('http://localhost:11434/api/tags')
            fetch_success = len(models) > 0
        elif provider == 'anthropic':
            # Anthropic doesn't have a models endpoint, return hardcoded
            models = [
                {'id': 'claude-sonnet-4-20250514', 'name': 'Claude Sonnet 4'},
                {'id': 'claude-3-5-sonnet-20241022', 'name': 'Claude 3.5 Sonnet'},
                {'id': 'claude-3-opus-20240229', 'name': 'Claude 3 Opus'},
                {'id': 'claude-3-haiku-20240307', 'name': 'Claude 3 Haiku'}
            ]
            fetch_success = True
        
        # Update cache only on successful fetch
        if fetch_success and len(models) > 0:
            if '_cached_models' not in config:
                config['_cached_models'] = {}
            config['_cached_models'][provider] = {
                'models': models,
                'cached_at': time.time()
            }
            save_config(config)
            logger.info(f"Cached {len(models)} models for {provider}")
        elif len(cached_list) > 0:
            # Fetch failed but we have cached data - use it
            logger.warning(f"Model fetch failed for {provider}, using stale cache")
            models = cached_list
        
        return jsonify({'success': True, 'models': models, 'cached': False})
        
    except Exception as e:
        logger.error(f"Error fetching models for {provider}: {e}")
        # Try to return cached models on error
        config = load_config()
        cached = config.get('_cached_models', {}).get(provider, {}).get('models', [])
        if cached:
            return jsonify({'success': True, 'models': cached, 'cached': True, 'error': str(e)})
        return jsonify({'success': False, 'error': str(e), 'models': []}), 500


def fetch_openai_compatible_models(url: str, auth: str) -> list:
    """Fetch models from an OpenAI-compatible API."""
    try:
        headers = {'Authorization': f'Bearer {auth}'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        models = []
        
        for model in data.get('data', []):
            model_id = model.get('id', '')
            # Filter to relevant models (skip embeddings, etc.)
            if any(skip in model_id.lower() for skip in ['embed', 'whisper', 'tts', 'dall-e', 'moderation']):
                continue
            models.append({
                'id': model_id,
                'name': model_id.replace('-', ' ').title()
            })
        
        # Sort by name
        models.sort(key=lambda x: x['name'])
        logger.info(f"Fetched {len(models)} models from {url}")
        return models
        
    except Exception as e:
        logger.error(f"Failed to fetch models from {url}: {e}")
        return []


def fetch_ollama_models(url: str) -> list:
    """Fetch models from local Ollama instance."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        models = []
        
        for model in data.get('models', []):
            name = model.get('name', '')
            models.append({
                'id': name,
                'name': name
            })
        
        logger.info(f"Fetched {len(models)} models from Ollama")
        return models
        
    except Exception as e:
        logger.warning(f"Failed to fetch Ollama models (is Ollama running?): {e}")
        return []


@app.route('/api/voices/<provider>')
def get_provider_voices(provider):
    """Get available voices for a specific provider."""
    try:
        if provider in PROVIDER_VOICES:
            voices = [{'id': v, 'name': v} for v in PROVIDER_VOICES[provider]]
            return jsonify({'success': True, 'voices': voices})
        
        if provider == 'edge_tts':
            # Edge TTS has many voices, fetch async
            voices = asyncio.run(get_edge_tts_voices())
            return jsonify({'success': True, 'voices': voices})
        
        if provider == 'elevenlabs':
            # Would need API call to fetch, return placeholder
            return jsonify({
                'success': True, 
                'voices': [{'id': 'fetch_from_api', 'name': 'Configure API key to see voices'}]
            })
        
        return jsonify({'success': True, 'voices': []})
        
    except Exception as e:
        logger.error(f"Error fetching voices for {provider}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


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
            if provider == 'local_whisper':
                transcription = transcribe_with_local_whisper(tmp_path)
            elif provider in ['whisper', 'openai_whisper']:
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
        
        logger.info(f"Using inference: provider={provider}, model={model}")
        
        # Build messages array
        messages = []
        for msg in conversation_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            messages.append({'role': role, 'content': content})
        
        # Add current message
        messages.append({'role': 'user', 'content': text})
        
        # Call the appropriate API
        # Normalize protocol names (grok, openai, ollama all use OpenAI-compatible format)
        openai_compatible_protocols = ['openai_compatible', 'grok', 'openai', 'ollama', 'xai_realtime']
        
        if protocol in openai_compatible_protocols:
            response_text = call_openai_compatible(url, auth, model, messages)
        elif protocol in ['anthropic_messages', 'anthropic']:
            response_text = call_anthropic(url, auth, model, messages)
        elif protocol == 'browser_speech_api':
            # This shouldn't be called server-side
            return jsonify({'error': 'Browser Speech API should be used client-side'}), 400
        else:
            logger.warning(f"Unknown protocol '{protocol}', trying OpenAI-compatible")
            response_text = call_openai_compatible(url, auth, model, messages)
        
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


def call_openai_compatible(url: str, auth: str, model: str, messages: list, enable_tools: bool = True) -> str:
    """Call an OpenAI-compatible API (Grok, OpenAI, Ollama, etc.) with tool support."""
    logger.debug(f"Calling OpenAI-compatible API: {url}, model: {model}")
    
    headers = {
        'Authorization': f'Bearer {auth}',
        'Content-Type': 'application/json'
    }
    
    # Define available tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "description": "Get the current date and time. Use this when the user asks about today's date, current time, or any time-sensitive information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Optional timezone (e.g., 'America/New_York', 'UTC'). Defaults to server local time."
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_x",
                "description": "Search X (formerly Twitter) for posts, news, and discussions. Use this when the user asks about recent events, trending topics, what people are saying, or needs real-time information from social media.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for X"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information. Use this when the user asks about facts, news, or information that may require looking up current data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for the web"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ] if enable_tools else None
    
    payload = {
        'model': model,
        'messages': messages.copy()
    }
    
    if tools:
        payload['tools'] = tools
        payload['tool_choice'] = 'auto'
    
    # Tool loop - keep calling until we get a final response
    max_tool_iterations = 5
    for iteration in range(max_tool_iterations):
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        choice = data['choices'][0]
        message = choice['message']
        
        # Check if there are tool calls
        tool_calls = message.get('tool_calls', [])
        
        if not tool_calls:
            # No tool calls, return the content
            return message.get('content', '')
        
        logger.info(f"Tool calls requested (iteration {iteration + 1}): {[tc['function']['name'] for tc in tool_calls]}")
        
        # Add assistant message with tool calls to conversation
        payload['messages'].append(message)
        
        # Execute each tool call
        for tool_call in tool_calls:
            function_name = tool_call['function']['name']
            function_args = json.loads(tool_call['function']['arguments'])
            tool_call_id = tool_call['id']
            
            logger.debug(f"Executing tool: {function_name} with args: {function_args}")
            
            # Execute the tool
            try:
                result = execute_tool(function_name, function_args, auth)
                logger.info(f"Tool {function_name} result: {str(result)[:200]}...")
            except Exception as e:
                logger.error(f"Tool {function_name} error: {e}")
                result = f"Error executing {function_name}: {str(e)}"
            
            # Add tool result to conversation
            payload['messages'].append({
                'role': 'tool',
                'tool_call_id': tool_call_id,
                'content': str(result)
            })
    
    # If we hit max iterations, return whatever we have
    logger.warning(f"Hit max tool iterations ({max_tool_iterations})")
    return message.get('content', 'I was unable to complete the request after multiple tool calls.')


def execute_tool(function_name: str, args: dict, auth: str) -> str:
    """Execute a tool and return the result."""
    if function_name == 'get_current_datetime':
        return tool_get_current_datetime(args.get('timezone'))
    elif function_name == 'search_x':
        return tool_search_x(args.get('query', ''), auth)
    elif function_name == 'search_web':
        return tool_search_web(args.get('query', ''), auth)
    else:
        return f"Unknown tool: {function_name}"


def tool_get_current_datetime(timezone: str = None) -> str:
    """Get the current date and time."""
    from datetime import datetime
    import pytz
    
    try:
        if timezone:
            tz = pytz.timezone(timezone)
            now = datetime.now(tz)
        else:
            now = datetime.now()
        
        return now.strftime("%A, %B %d, %Y at %I:%M %p %Z").strip()
    except Exception as e:
        # Fallback to simple format
        from datetime import datetime
        return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


def tool_search_x(query: str, auth: str) -> str:
    """Search X (Twitter) using the xAI Agent Tools API."""
    if not query:
        return "No search query provided"
    
    logger.info(f"Searching X for: {query}")
    
    try:
        # Use xAI Agent Tools API
        headers = {
            'Authorization': f'Bearer {auth}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'query': query,
            'source': 'x'  # Specify X/Twitter as the source
        }
        
        response = requests.post(
            'https://api.x.ai/v1/agent-tools/search',
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            # Format the results
            results = data.get('results', [])
            if results:
                formatted = []
                for i, result in enumerate(results[:10], 1):  # Top 10 results
                    text = result.get('text', result.get('content', ''))
                    author = result.get('author', result.get('user', {}).get('name', 'Unknown'))
                    formatted.append(f"{i}. @{author}: {text[:280]}")
                return "\n\n".join(formatted)
            return "No results found on X for this query."
        else:
            logger.warning(f"X search returned {response.status_code}: {response.text[:200]}")
            return f"X search unavailable (status {response.status_code}). Try asking the question directly."
            
    except Exception as e:
        logger.error(f"X search error: {e}")
        return f"X search error: {str(e)}"


def tool_search_web(query: str, auth: str) -> str:
    """Search the web using the xAI Agent Tools API."""
    if not query:
        return "No search query provided"
    
    logger.info(f"Searching web for: {query}")
    
    try:
        # Use xAI Agent Tools API
        headers = {
            'Authorization': f'Bearer {auth}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'query': query,
            'source': 'web'
        }
        
        response = requests.post(
            'https://api.x.ai/v1/agent-tools/search',
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            # Format the results
            results = data.get('results', [])
            if results:
                formatted = []
                for i, result in enumerate(results[:5], 1):  # Top 5 results
                    title = result.get('title', 'Untitled')
                    snippet = result.get('snippet', result.get('content', ''))[:300]
                    url = result.get('url', '')
                    formatted.append(f"{i}. {title}\n   {snippet}\n   {url}")
                return "\n\n".join(formatted)
            return "No web results found for this query."
        else:
            logger.warning(f"Web search returned {response.status_code}: {response.text[:200]}")
            return f"Web search unavailable (status {response.status_code}). Try asking the question directly."
            
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Web search error: {str(e)}"


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
        voice = api_config.get('voice', 'Ara')
        
        if provider == 'browser':
            # Return text for client-side synthesis
            return jsonify({
                'success': True,
                'use_browser': True,
                'text': text
            })
        
        if provider == 'edge_tts':
            # Use Edge TTS
            audio_data = asyncio.run(synthesize_with_edge_tts(text, voice))
            if audio_data:
                import base64
                return jsonify({
                    'success': True,
                    'audio_base64': base64.b64encode(audio_data).decode('utf-8'),
                    'audio_format': 'mp3'
                })
            else:
                return jsonify({'success': False, 'error': 'Edge TTS synthesis failed'}), 500
        
        if provider == 'openai_tts':
            url = api_config.get('url', 'https://api.openai.com/v1/audio/speech')
            auth = api_config.get('auth', '')
            audio_data = synthesize_with_openai_tts(url, auth, text, voice)
            if audio_data:
                import base64
                return jsonify({
                    'success': True,
                    'audio_base64': base64.b64encode(audio_data).decode('utf-8'),
                    'audio_format': 'mp3'
                })
            else:
                return jsonify({'success': False, 'error': 'OpenAI TTS synthesis failed'}), 500
        
        # For other providers, fallback to browser
        logger.warning(f"Server-side synthesis for {provider} not implemented, using browser")
        return jsonify({
            'success': True,
            'use_browser': True,
            'text': text
        })
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


async def synthesize_with_edge_tts(text: str, voice: str) -> bytes:
    """Synthesize speech using Edge TTS."""
    import edge_tts
    import io
    
    logger.debug(f"Edge TTS synthesis: voice={voice}, text={text[:50]}...")
    
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_data = io.BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        
        return audio_data.getvalue()
    except Exception as e:
        logger.error(f"Edge TTS error: {e}")
        return None


def synthesize_with_openai_tts(url: str, auth: str, text: str, voice: str) -> bytes:
    """Synthesize speech using OpenAI TTS API."""
    logger.debug(f"OpenAI TTS synthesis: voice={voice}, text={text[:50]}...")
    
    headers = {
        'Authorization': f'Bearer {auth}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': 'tts-1',
        'input': text,
        'voice': voice
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"OpenAI TTS error: {e}")
        return None


if __name__ == '__main__':
    # API keys are now configured via settings panel, no env var needed
    logger.info("Starting Groklexa server...")
    
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
        logger.info("Starting HTTPS web server...")
        logger.info("Open https://spark.home.arpa:5001 in your browser")
        logger.info("Note: Your browser will show a security warning for the self-signed certificate.")
        logger.info("      Click 'Advanced' and 'Proceed' to accept it.")
    else:
        logger.info("Starting HTTP web server (no SSL certificate found)...")
        logger.info("Run './tools/generate_cert.sh' to generate a certificate for HTTPS support.")
        logger.info("Open http://localhost:5001 in your browser")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5001,
        ssl_context=ssl_context
    )

