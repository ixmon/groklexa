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
import threading
import uuid
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


def get_default_persona():
    """Get the default persona configuration."""
    return {
        "name": "Groklexa",
        "prompt": "You are Groklexa, a friendly and witty voice assistant. You speak naturally and conversationally, like a helpful friend who happens to know a lot about everything.",
        "mode": "single",
        "single": {
            "provider": "xai_realtime",
            "url": "wss://api.x.ai/v1/realtime",
            "auth": "",
            "voice": "Ara",
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
            "auth": "",
            "protocol": "openai_compatible",
            "model": "grok-3"
        },
        "synthesis": {
            "provider": "grok",
            "url": "wss://api.x.ai/v1/realtime",
            "auth": "",
            "protocol": "xai_realtime",
            "voice": "Ara"
        },
        "tools": {
            "get_current_datetime": True,
            "get_current_weather": True,
            "set_timer": True,
            "search_x": True,
            "search_web": True
        }
    }


def migrate_legacy_config(config):
    """Migrate legacy config (without personas) to new persona-based format."""
    if 'personas' in config:
        return config  # Already migrated
    
    # Create a default persona from existing settings
    persona = get_default_persona()
    persona['name'] = 'Groklexa'
    
    # Copy existing settings into the persona
    for key in ['mode', 'single', 'transcription', 'inference', 'synthesis']:
        if key in config:
            persona[key] = config[key]
    
    # Preserve any existing prompt
    if 'prompt' in config:
        persona['prompt'] = config['prompt']
    
    # Preserve model cache if present
    model_cache = config.get('_model_cache', {})
    
    return {
        'active_persona': 'default',
        'personas': {
            'default': persona
        },
        '_model_cache': model_cache
    }


def load_config():
    """Load API configuration from file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return migrate_legacy_config(config)
    elif CONFIG_EXAMPLE.exists():
        with open(CONFIG_EXAMPLE, 'r') as f:
            config = json.load(f)
            return migrate_legacy_config(config)
    else:
        # Default config with personas
        return {
            'active_persona': 'default',
            'personas': {
                'default': get_default_persona()
            },
            '_model_cache': {}
        }


def get_active_persona(config=None):
    """Get the currently active persona configuration."""
    if config is None:
        config = load_config()
    
    active_id = config.get('active_persona', 'default')
    personas = config.get('personas', {})
    
    if active_id in personas:
        persona = personas[active_id]
        persona['_id'] = active_id  # Include the ID
        return persona
    
    # Fallback to first persona or default
    if personas:
        first_id = list(personas.keys())[0]
        persona = personas[first_id]
        persona['_id'] = first_id
        return persona
    
    return get_default_persona()


def save_config(config):
    """Save API configuration to file."""
    CONFIG_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def mask_auth(config):
    """Mask auth strings for safe transmission to frontend."""
    masked = json.loads(json.dumps(config))  # Deep copy
    
    # Handle persona-based config
    if 'personas' in masked:
        for persona_id, persona in masked['personas'].items():
            for section in ['single', 'transcription', 'inference', 'synthesis']:
                if section in persona and 'auth' in persona[section]:
                    if persona[section]['auth']:
                        persona[section]['auth'] = AUTH_MASK
    else:
        # Legacy format
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
    
    # Get active persona for convenience
    active_persona = get_active_persona(config)
    
    # Add detected models to active persona
    for section in ['single', 'transcription', 'inference', 'synthesis']:
        if section in active_persona and 'url' in active_persona[section]:
            detected = detect_model_from_url(active_persona[section]['url'])
            if detected:
                if 'personas' in masked:
                    active_id = config.get('active_persona', 'default')
                    if active_id in masked['personas']:
                        masked['personas'][active_id][section]['detected_model'] = detected
    
    # Also include a flattened view of active persona for backward compatibility
    active_id = config.get('active_persona', 'default')
    if 'personas' in masked and active_id in masked['personas']:
        active = masked['personas'][active_id].copy()
        active['_id'] = active_id
        masked['active'] = active
    
    return jsonify({
        'success': True,
        'config': masked
    })


@app.route('/api/config', methods=['POST'])
def set_config():
    """Save API configuration (supports both legacy and persona-based)."""
    try:
        new_config = request.json
        if not new_config:
            return jsonify({'success': False, 'error': 'No configuration provided'}), 400
        
        # Load existing config to preserve unchanged auth strings
        existing_config = load_config()
        
        # Check if this is a persona-based config
        if 'personas' in new_config:
            # New persona-based format
            for persona_id, persona in new_config['personas'].items():
                existing_persona = existing_config.get('personas', {}).get(persona_id, {})
                for section in ['single', 'transcription', 'inference', 'synthesis']:
                    if section in persona and 'auth' in persona[section]:
                        if persona[section]['auth'] in [AUTH_UNCHANGED, AUTH_MASK, '']:
                            if section in existing_persona and 'auth' in existing_persona[section]:
                                persona[section]['auth'] = existing_persona[section]['auth']
            
            # Preserve model cache
            if '_model_cache' in existing_config:
                new_config['_model_cache'] = existing_config['_model_cache']
        else:
            # Legacy format - convert to persona format
            for section in ['single', 'transcription', 'inference', 'synthesis']:
                if section in new_config and 'auth' in new_config[section]:
                    if new_config[section]['auth'] in [AUTH_UNCHANGED, AUTH_MASK, '']:
                        active_persona = get_active_persona(existing_config)
                        if section in active_persona and 'auth' in active_persona[section]:
                            new_config[section]['auth'] = active_persona[section]['auth']
            
            # Wrap in persona format
            active_id = existing_config.get('active_persona', 'default')
            personas = existing_config.get('personas', {})
            
            # Update the active persona with new settings
            if active_id in personas:
                for key in ['mode', 'single', 'transcription', 'inference', 'synthesis', 'prompt']:
                    if key in new_config:
                        personas[active_id][key] = new_config[key]
            else:
                personas[active_id] = new_config
            
            new_config = {
                'active_persona': active_id,
                'personas': personas,
                '_model_cache': existing_config.get('_model_cache', {})
            }
        
        save_config(new_config)
        
        return jsonify({
            'success': True,
            'message': 'Configuration saved'
        })
    except Exception as e:
        logger.error(f"Error saving config: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ========== PERSONA MANAGEMENT ==========

@app.route('/api/personas', methods=['GET'])
def list_personas():
    """List all available personas."""
    config = load_config()
    personas = config.get('personas', {})
    active_id = config.get('active_persona', 'default')
    
    # Return list with masked auth
    persona_list = []
    for pid, persona in personas.items():
        persona_list.append({
            'id': pid,
            'name': persona.get('name', pid),
            'prompt': persona.get('prompt', '')[:100] + '...' if len(persona.get('prompt', '')) > 100 else persona.get('prompt', ''),
            'active': pid == active_id
        })
    
    return jsonify({
        'success': True,
        'personas': persona_list,
        'active_persona': active_id
    })


@app.route('/api/personas/switch', methods=['POST'])
def switch_persona():
    """Switch to a different persona."""
    try:
        data = request.json
        persona_id = data.get('persona_id')
        
        if not persona_id:
            return jsonify({'success': False, 'error': 'No persona_id provided'}), 400
        
        config = load_config()
        
        if persona_id not in config.get('personas', {}):
            return jsonify({'success': False, 'error': f'Persona {persona_id} not found'}), 404
        
        config['active_persona'] = persona_id
        save_config(config)
        
        # Return the full active persona config (masked)
        masked = mask_auth(config)
        active = masked['personas'][persona_id].copy()
        active['_id'] = persona_id
        
        return jsonify({
            'success': True,
            'message': f'Switched to persona: {active.get("name", persona_id)}',
            'active': active
        })
    except Exception as e:
        logger.error(f"Error switching persona: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/personas', methods=['POST'])
def create_persona():
    """Create a new persona."""
    try:
        data = request.json
        persona_id = data.get('id')
        persona_data = data.get('persona', {})
        
        if not persona_id:
            # Generate ID from name
            name = persona_data.get('name', 'New Persona')
            persona_id = name.lower().replace(' ', '_').replace('-', '_')
            # Make unique
            import time
            persona_id = f"{persona_id}_{int(time.time()) % 10000}"
        
        config = load_config()
        
        # Start with default and merge provided data
        new_persona = get_default_persona()
        for key in persona_data:
            if key in new_persona:
                if isinstance(new_persona[key], dict) and isinstance(persona_data[key], dict):
                    new_persona[key].update(persona_data[key])
                else:
                    new_persona[key] = persona_data[key]
        
        config['personas'][persona_id] = new_persona
        save_config(config)
        
        return jsonify({
            'success': True,
            'message': f'Created persona: {new_persona.get("name", persona_id)}',
            'persona_id': persona_id
        })
    except Exception as e:
        logger.error(f"Error creating persona: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/personas/<persona_id>', methods=['PUT'])
def update_persona(persona_id):
    """Update an existing persona."""
    try:
        data = request.json
        
        config = load_config()
        
        if persona_id not in config.get('personas', {}):
            return jsonify({'success': False, 'error': f'Persona {persona_id} not found'}), 404
        
        existing = config['personas'][persona_id]
        
        # Update fields
        for key in data:
            if key.startswith('_'):
                continue  # Skip internal fields
            if key in existing and isinstance(existing[key], dict) and isinstance(data[key], dict):
                # Handle auth masking for nested dicts
                if 'auth' in data[key] and data[key]['auth'] in [AUTH_UNCHANGED, AUTH_MASK]:
                    data[key]['auth'] = existing[key].get('auth', '')
                existing[key].update(data[key])
            else:
                # Simple value update or new key
                existing[key] = data[key]
        
        save_config(config)
        
        return jsonify({
            'success': True,
            'message': f'Updated persona: {existing.get("name", persona_id)}'
        })
    except Exception as e:
        logger.error(f"Error updating persona: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/personas/<persona_id>', methods=['DELETE'])
def delete_persona(persona_id):
    """Delete a persona."""
    try:
        config = load_config()
        
        if persona_id not in config.get('personas', {}):
            return jsonify({'success': False, 'error': f'Persona {persona_id} not found'}), 404
        
        # Can't delete the last persona
        if len(config['personas']) <= 1:
            return jsonify({'success': False, 'error': 'Cannot delete the last persona'}), 400
        
        # Can't delete active persona without switching first
        if config.get('active_persona') == persona_id:
            # Switch to another persona
            other_id = [k for k in config['personas'].keys() if k != persona_id][0]
            config['active_persona'] = other_id
        
        del config['personas'][persona_id]
        save_config(config)
        
        return jsonify({
            'success': True,
            'message': f'Deleted persona: {persona_id}',
            'active_persona': config['active_persona']
        })
    except Exception as e:
        logger.error(f"Error deleting persona: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/personas/<persona_id>/duplicate', methods=['POST'])
def duplicate_persona(persona_id):
    """Duplicate an existing persona."""
    try:
        config = load_config()
        
        if persona_id not in config.get('personas', {}):
            return jsonify({'success': False, 'error': f'Persona {persona_id} not found'}), 404
        
        # Deep copy the persona
        original = config['personas'][persona_id]
        new_persona = json.loads(json.dumps(original))
        new_persona['name'] = f"{original.get('name', persona_id)} (Copy)"
        
        # Generate new ID
        import time
        new_id = f"{persona_id}_copy_{int(time.time()) % 10000}"
        
        config['personas'][new_id] = new_persona
        save_config(config)
        
        return jsonify({
            'success': True,
            'message': f'Duplicated persona as: {new_persona["name"]}',
            'persona_id': new_id
        })
    except Exception as e:
        logger.error(f"Error duplicating persona: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


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
# Using OpenAI's whisper package (PyTorch-based) for CUDA support on aarch64
_whisper_model = None
_whisper_model_size = 'base'  # Options: tiny, base, small, medium, large


def get_whisper_model():
    """Get or initialize the local Whisper model (OpenAI whisper with PyTorch/CUDA)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    
    try:
        import torch
        import whisper
        
        # Use CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model: {_whisper_model_size} on {device}")
        
        _whisper_model = whisper.load_model(_whisper_model_size, device=device)
        logger.info(f"Whisper model loaded successfully on {device}")
        return _whisper_model
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return None


def transcribe_with_local_whisper(audio_path: str) -> str:
    """Transcribe audio using local Whisper model (OpenAI whisper with PyTorch)."""
    import time
    
    t0 = time.time()
    model = get_whisper_model()
    t1 = time.time()
    
    if model is None:
        raise Exception("Whisper model not available")
    
    logger.debug(f"Transcribing with local Whisper: {audio_path} (model fetch: {(t1-t0)*1000:.0f}ms)")
    
    try:
        # OpenAI whisper returns a dict with 'text' key
        t2 = time.time()
        result = model.transcribe(audio_path, language="en")
        t3 = time.time()
        
        transcript = result.get("text", "")
        logger.info(f"Local Whisper transcription ({(t3-t2)*1000:.0f}ms): {transcript[:100]}...")
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
        active_persona = get_active_persona(config)
        mode = active_persona.get('mode', 'separate')
        
        if mode == 'single':
            api_config = active_persona.get('single', {})
        else:
            api_config = active_persona.get('inference', {})
        
        auth = api_config.get('auth', '') or os.environ.get('XAI_API_KEY', '')
        
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
        
        # Load config from active persona
        config = load_config()
        active_persona = get_active_persona(config)
        mode = active_persona.get('mode', 'separate')
        
        if mode == 'single':
            api_config = active_persona.get('single', {})
        else:
            api_config = active_persona.get('transcription', {})
        
        provider = api_config.get('provider', 'browser')
        logger.info(f"Transcription provider: {provider}")
        
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


def get_system_prompt(synthesis_provider: str, custom_prompt: str = None) -> str:
    """Generate a system prompt tailored to the TTS provider being used."""
    
    # Use custom prompt from persona if provided, otherwise default
    if custom_prompt:
        base_prompt = custom_prompt
    else:
        base_prompt = """You are Groklexa, a friendly and witty voice assistant. You speak naturally and conversationally, like a helpful friend who happens to know a lot about everything."""
    
    # TTS-specific formatting guidance
    tts_guidance = {
        'chatterbox': """
Your responses will be spoken aloud using Chatterbox voice cloning. For best results:
- Write naturally as if speaking - contractions, casual phrasing, natural pauses
- Avoid special characters, emojis, or formatting that won't speak well
- Use punctuation for pacing: commas for brief pauses, periods for longer pauses
- Spell out abbreviations and numbers (say "twenty-five" not "25")
- Keep responses concise - aim for 1-3 sentences unless more detail is needed
- Express emotion through word choice, not symbols""",
        
        'edge_tts': """
Your responses will be spoken aloud using Edge TTS. For best results:
- Write naturally as if speaking - contractions, casual phrasing
- Avoid emojis and special characters
- Use punctuation for pacing
- Keep responses conversational and concise
- Spell out abbreviations when natural to do so""",
        
        'browser': """
Your responses will be spoken aloud using browser text-to-speech. For best results:
- Keep sentences short and clear
- Avoid special characters and emojis
- Use simple punctuation for natural pacing
- Be concise - browser TTS works best with shorter responses""",
        
        'openai_tts': """
Your responses will be spoken aloud using OpenAI TTS. For best results:
- Write naturally and conversationally
- Use punctuation for expression and pacing
- Avoid excessive formatting or special characters
- You can be more expressive as OpenAI TTS handles nuance well""",
        
        'elevenlabs': """
Your responses will be spoken aloud using ElevenLabs. For best results:
- Write naturally with good emotional expression
- Use punctuation creatively for pacing and emphasis
- Avoid special characters but express emotion through words
- ElevenLabs handles nuance well, so be expressive""",
    }
    
    # Default guidance for unknown providers
    default_guidance = """
Your responses will be spoken aloud. For best results:
- Write naturally as if speaking
- Avoid emojis and special formatting
- Use punctuation for pacing
- Keep responses concise and conversational"""
    
    guidance = tts_guidance.get(synthesis_provider, default_guidance)
    
    return f"{base_prompt}\n{guidance}"


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
        
        # Load active persona configuration
        config = load_config()
        persona = get_active_persona(config)
        mode = persona.get('mode', 'single')
        
        if mode == 'single':
            # Use single API config
            api_config = persona.get('single', {})
            synth_config = persona.get('single', {})
        else:
            # Use separate configs
            api_config = persona.get('inference', {})
            synth_config = persona.get('synthesis', {})
        
        provider = api_config.get('provider', 'grok')
        url = api_config.get('url', '')
        auth = api_config.get('auth', '')
        protocol = api_config.get('protocol', 'openai_compatible')
        model = api_config.get('model', 'grok-3')
        
        # Get synthesis provider and persona's custom prompt
        synth_provider = synth_config.get('provider', 'browser')
        custom_prompt = persona.get('prompt', None)
        
        # Get persona's tool permissions
        tool_permissions = persona.get('tools', {})
        
        logger.info(f"Using inference: provider={provider}, model={model}, synth={synth_provider}, persona={persona.get('name', 'default')}, tools={list(k for k,v in tool_permissions.items() if v)}")
        
        # Build messages array with system prompt (using persona's custom prompt)
        system_prompt = get_system_prompt(synth_provider, custom_prompt)
        messages = [{'role': 'system', 'content': system_prompt}]
        
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
            # Pass persona's tool permissions to the API call
            response_text = call_openai_compatible(url, auth, model, messages, tool_permissions=tool_permissions)
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


def call_openai_compatible(url: str, auth: str, model: str, messages: list, tool_permissions: dict = None) -> str:
    """Call an OpenAI-compatible API (Grok, OpenAI, Ollama, etc.) with tool support."""
    logger.debug(f"Calling OpenAI-compatible API: {url}, model: {model}")
    
    headers = {
        'Authorization': f'Bearer {auth}',
        'Content-Type': 'application/json'
    }
    
    # Define all available tools
    all_tools = {
        "get_current_datetime": {
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
        "search_x": {
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
        "search_web": {
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
        },
        "get_current_weather": {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather and forecast for a location. Use this when the user asks about weather, temperature, rain, snow, or outdoor conditions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, optionally with country code (e.g., 'London', 'Paris,FR', 'New York,US')"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        "set_timer": {
            "type": "function",
            "function": {
                "name": "set_timer",
                "description": "Set a timer or reminder. Use this when the user asks to be reminded in X minutes, set a timer, or wants an alert after a duration.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "minutes": {
                            "type": "number",
                            "description": "Duration in minutes (can be decimal, e.g., 0.5 for 30 seconds)"
                        },
                        "message": {
                            "type": "string",
                            "description": "What to remind the user about (e.g., 'check the oven', 'take a break', 'meeting starts')"
                        }
                    },
                    "required": ["minutes"]
                }
            }
        }
    }
    
    # Filter tools based on persona permissions
    tools = None
    if tool_permissions:
        enabled_tools = [all_tools[name] for name, enabled in tool_permissions.items() if enabled and name in all_tools]
        if enabled_tools:
            tools = enabled_tools
            logger.debug(f"Enabled tools: {[t['function']['name'] for t in tools]}")
    
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
        
        # Check if model doesn't support tools - retry without them
        if response.status_code == 400 and tools:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', '')
                if 'does not support tools' in error_msg:
                    logger.warning(f"Model doesn't support tools, retrying without: {error_msg}")
                    del payload['tools']
                    del payload['tool_choice']
                    tools = None
                    response = requests.post(url, headers=headers, json=payload, timeout=120)
            except:
                pass
        
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
    elif function_name == 'get_current_weather':
        return tool_get_current_weather(args.get('location', ''))
    elif function_name == 'set_timer':
        return tool_set_timer(args.get('minutes', 1), args.get('message', ''))
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


# Weather cache: {location: {'data': ..., 'timestamp': ...}}
_weather_cache = {}
_WEATHER_CACHE_DURATION = 30 * 60  # 30 minutes

# Timer storage: list of active timers
_active_timers = []
_timer_lock = threading.Lock()

def tool_get_current_weather(location: str) -> str:
    """Get current weather using OpenWeatherMap API with caching."""
    import time
    
    if not location:
        return "No location provided. Please specify a city name."
    
    # Check cache first
    cache_key = location.lower().strip()
    cached = _weather_cache.get(cache_key)
    if cached:
        age = time.time() - cached['timestamp']
        if age < _WEATHER_CACHE_DURATION:
            logger.debug(f"Weather cache hit for {location} (age: {age:.0f}s)")
            return cached['data']
    
    # Get API key from environment
    api_key = os.environ.get('OPENWEATHERMAP_API_KEY', '')
    if not api_key:
        return f"Weather service not configured. Set OPENWEATHERMAP_API_KEY environment variable. (Location requested: {location})"
    
    try:
        # OpenWeatherMap API call
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': location,
            'appid': api_key,
            'units': 'imperial'  # Use Fahrenheit
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 404:
            return f"Location '{location}' not found. Try a different city name."
        
        response.raise_for_status()
        data = response.json()
        
        # Extract weather info
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        city_name = data['name']
        country = data['sys'].get('country', '')
        wind_speed = data['wind']['speed']
        
        # Format response
        result = f"""Weather in {city_name}, {country}:
- Conditions: {description.title()}
- Temperature: {temp:.0f}°F (feels like {feels_like:.0f}°F)
- Humidity: {humidity}%
- Wind: {wind_speed:.0f} mph"""
        
        # Cache the result
        _weather_cache[cache_key] = {
            'data': result,
            'timestamp': time.time()
        }
        
        logger.info(f"Weather for {location}: {temp:.0f}°F, {description}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API error: {e}")
        return f"Unable to fetch weather for {location}. Please try again later."
    except Exception as e:
        logger.error(f"Weather tool error: {e}")
        return f"Error getting weather: {str(e)}"


def tool_set_timer(minutes: float, message: str = '') -> str:
    """Set a timer that will fire after the specified duration."""
    import time
    
    if minutes <= 0:
        return "Timer duration must be positive."
    
    if minutes > 1440:  # 24 hours max
        return "Timer duration cannot exceed 24 hours."
    
    seconds = int(minutes * 60)
    timer_id = str(uuid.uuid4())[:8]
    expires_at = time.time() + seconds
    
    timer = {
        'id': timer_id,
        'minutes': minutes,
        'seconds': seconds,
        'message': message or 'Timer complete',
        'expires_at': expires_at,
        'status': 'active',
        'created_at': time.time()
    }
    
    with _timer_lock:
        _active_timers.append(timer)
    
    # Format duration for response
    if minutes >= 60:
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        duration_str = f"{hours} hour{'s' if hours != 1 else ''}"
        if mins > 0:
            duration_str += f" and {mins} minute{'s' if mins != 1 else ''}"
    elif minutes >= 1:
        duration_str = f"{int(minutes)} minute{'s' if minutes != 1 else ''}"
    else:
        duration_str = f"{seconds} second{'s' if seconds != 1 else ''}"
    
    if message:
        result = f"Timer set for {duration_str}. I'll remind you: {message}"
    else:
        result = f"Timer set for {duration_str}."
    
    logger.info(f"Timer {timer_id} set: {duration_str}, message: '{message}'")
    return result


def get_fired_timers() -> list:
    """Get list of timers that have fired and mark them as acknowledged."""
    import time
    
    fired = []
    current_time = time.time()
    
    with _timer_lock:
        for timer in _active_timers:
            if timer['status'] == 'active' and current_time >= timer['expires_at']:
                timer['status'] = 'fired'
                fired.append(timer.copy())
        
        # Clean up old acknowledged timers (keep for 1 minute after firing)
        _active_timers[:] = [t for t in _active_timers 
                            if t['status'] == 'active' or 
                            (t['status'] == 'fired' and current_time - t['expires_at'] < 60)]
    
    return fired


def acknowledge_timer(timer_id: str):
    """Mark a timer as acknowledged."""
    with _timer_lock:
        for timer in _active_timers:
            if timer['id'] == timer_id:
                timer['status'] = 'acknowledged'
                break


@app.route('/api/timers', methods=['GET'])
def get_timers():
    """Get all active timers and any that have fired."""
    import time
    
    fired = get_fired_timers()
    current_time = time.time()
    
    with _timer_lock:
        active = [t.copy() for t in _active_timers if t['status'] == 'active']
        # Add remaining time to active timers
        for t in active:
            t['remaining_seconds'] = max(0, int(t['expires_at'] - current_time))
    
    return jsonify({
        'success': True,
        'active': active,
        'fired': fired
    })


@app.route('/api/timers/<timer_id>/acknowledge', methods=['POST'])
def ack_timer(timer_id):
    """Acknowledge a fired timer."""
    acknowledge_timer(timer_id)
    return jsonify({'success': True})


def tool_search_x(query: str, auth: str) -> str:
    """Search X (Twitter) using the xAI SDK."""
    if not query:
        return "No search query provided"
    
    logger.info(f"Searching X for: {query}")
    
    try:
        from xai_sdk import Client
        from xai_sdk.chat import user
        from xai_sdk.tools import x_search
        
        client = Client(api_key=auth)
        
        # Create a chat session with x_search tool
        # Note: Server-side tools require grok-4 family models
        chat = client.chat.create(
            model="grok-4-fast",
            tools=[x_search()],
        )
        
        # Ask for search results
        chat.append(user(f"Search X for: {query}. Return a summary of the top posts and discussions."))
        
        # Get the response (non-streaming)
        response = chat.sample()
        
        if response and response.content:
            logger.info(f"X search result: {response.content[:200]}...")
            return response.content
        else:
            return "No results found on X for this query."
            
    except ImportError as e:
        logger.error(f"xai-sdk import error: {e}")
        return "X search unavailable (xai-sdk not installed)"
    except Exception as e:
        logger.error(f"X search error: {e}", exc_info=True)
        return f"X search error: {str(e)}"


def tool_search_web(query: str, auth: str) -> str:
    """Search the web using the xAI SDK."""
    if not query:
        return "No search query provided"
    
    logger.info(f"Searching web for: {query}")
    
    try:
        from xai_sdk import Client
        from xai_sdk.chat import user
        from xai_sdk.tools import web_search
        
        client = Client(api_key=auth)
        
        # Create a chat session with web_search tool
        # Note: Server-side tools require grok-4 family models
        chat = client.chat.create(
            model="grok-4-fast",
            tools=[web_search()],
        )
        
        # Ask for search results
        chat.append(user(f"Search the web for: {query}. Return the key findings and information."))
        
        # Get the response (non-streaming)
        response = chat.sample()
        
        if response and response.content:
            logger.info(f"Web search result: {response.content[:200]}...")
            return response.content
        else:
            return "No web results found for this query."
            
    except ImportError as e:
        logger.error(f"xai-sdk import error: {e}")
        return "Web search unavailable (xai-sdk not installed)"
    except Exception as e:
        logger.error(f"Web search error: {e}", exc_info=True)
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
    
    response = requests.post(url, headers=headers, json=payload, timeout=120)
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
        
        # Load config from active persona
        config = load_config()
        active_persona = get_active_persona(config)
        mode = active_persona.get('mode', 'separate')
        
        if mode == 'single':
            api_config = active_persona.get('single', {})
        else:
            api_config = active_persona.get('synthesis', {})
        
        provider = api_config.get('provider', 'browser')
        voice = api_config.get('voice', 'Ara')
        
        logger.info(f"Synthesis provider: {provider}, voice: {voice}")
        
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
        
        if provider == 'chatterbox':
            # Use Chatterbox TTS with voice cloning
            audio_data = synthesize_with_chatterbox(text, voice)
            if audio_data:
                import base64
                return jsonify({
                    'success': True,
                    'audio_base64': base64.b64encode(audio_data).decode('utf-8'),
                    'audio_format': 'wav'
                })
            else:
                return jsonify({'success': False, 'error': 'Chatterbox synthesis failed. Check that voice reference exists.'}), 500
        
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


# Global Chatterbox model cache
_chatterbox_model = None
_chatterbox_model_loading = False


def get_chatterbox_model():
    """Get or load the Chatterbox TTS model (cached)."""
    global _chatterbox_model, _chatterbox_model_loading
    
    if _chatterbox_model is not None:
        return _chatterbox_model
    
    if _chatterbox_model_loading:
        # Wait for model to load
        import time
        for _ in range(60):  # Wait up to 60 seconds
            if _chatterbox_model is not None:
                return _chatterbox_model
            time.sleep(1)
        return None
    
    try:
        _chatterbox_model_loading = True
        logger.info("Loading Chatterbox TTS model...")
        
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        
        # Use CUDA if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Chatterbox using device: {device}")
        
        _chatterbox_model = ChatterboxTurboTTS.from_pretrained(device=device)
        logger.info("Chatterbox model loaded successfully")
        
        return _chatterbox_model
    except Exception as e:
        logger.error(f"Failed to load Chatterbox model: {e}", exc_info=True)
        return None
    finally:
        _chatterbox_model_loading = False


def synthesize_with_chatterbox(text: str, voice_ref: str) -> bytes:
    """Synthesize speech using Chatterbox TTS with voice cloning."""
    import io
    import time
    import numpy as np
    from scipy.io import wavfile
    
    t0 = time.time()
    logger.debug(f"Chatterbox synthesis: voice_ref={voice_ref}, text={text[:50]}...")
    
    try:
        t1 = time.time()
        model = get_chatterbox_model()
        t2 = time.time()
        
        if model is None:
            logger.error("Chatterbox model not available")
            return None
        
        logger.debug(f"Chatterbox model fetch: {(t2-t1)*1000:.0f}ms")
        
        # Build path to voice reference
        voice_path = os.path.join('static', 'voices', voice_ref)
        if not os.path.exists(voice_path):
            logger.error(f"Voice reference not found: {voice_path}")
            return None
        
        logger.info(f"Generating speech with Chatterbox (voice: {voice_ref})")
        
        # Generate audio
        t3 = time.time()
        wav = model.generate(text, audio_prompt_path=voice_path)
        t4 = time.time()
        logger.info(f"Chatterbox generation: {(t4-t3)*1000:.0f}ms")
        
        # Convert tensor to numpy and save as WAV using scipy
        audio_np = wav.squeeze().cpu().numpy()
        # Normalize to int16 range for WAV
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        audio_buffer = io.BytesIO()
        wavfile.write(audio_buffer, model.sr, audio_int16)
        audio_buffer.seek(0)
        
        logger.info(f"Chatterbox generated {len(audio_buffer.getvalue())} bytes")
        return audio_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Chatterbox synthesis error: {e}", exc_info=True)
        return None


@app.route('/api/voices/chatterbox', methods=['GET'])
def get_chatterbox_voices():
    """List available Chatterbox voice reference files."""
    try:
        voices_dir = os.path.join('static', 'voices')
        
        if not os.path.exists(voices_dir):
            os.makedirs(voices_dir, exist_ok=True)
            return jsonify({
                'success': True,
                'voices': [],
                'message': 'No voice references found. Add .wav files to static/voices/'
            })
        
        voices = []
        for f in os.listdir(voices_dir):
            if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                # Create a friendly name from filename
                name = os.path.splitext(f)[0].replace('_', ' ').replace('-', ' ').title()
                voices.append({
                    'id': f,
                    'name': name
                })
        
        voices.sort(key=lambda x: x['name'])
        
        return jsonify({
            'success': True,
            'voices': voices
        })
        
    except Exception as e:
        logger.error(f"Error listing Chatterbox voices: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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

