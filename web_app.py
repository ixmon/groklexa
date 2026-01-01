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
            "set_reminder": True,
            "list_timers": True,
            "cancel_timer": True,
            "get_system_info": True,
            "escalate_thinking": True,
            "search_x": True,
            "search_web": True
        },
        "escalation": {
            "enabled": True,
            "provider": "grok",
            "model": "grok-4-1-fast-reasoning",
            "url": "https://api.x.ai/v1/chat/completions",
            "auth": "",
            "max_context_turns": 3,
            "rate_limit_per_hour": 10
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
        
        # Retrieve and inject relevant RAG insights
        relevant_insights = retrieve_relevant_insights(text)
        if relevant_insights:
            insights_context = format_insights_for_context(relevant_insights)
            system_prompt += insights_context
            logger.info(f"Injected {len(relevant_insights)} RAG insights into context")
        
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
        
        # Get tool calls that were made during this inference
        tools_called = list(_current_inference_tools) if _current_inference_tools else []
        
        return jsonify({
            'success': True,
            'response': response_text,
            'tools_called': tools_called
        })
        
    except Exception as e:
        logger.error(f"Text inference error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Track tool calls for the current inference (for UI display)
_current_inference_tools = []

def call_openai_compatible(url: str, auth: str, model: str, messages: list, tool_permissions: dict = None) -> str:
    """Call an OpenAI-compatible API (Grok, OpenAI, Ollama, etc.) with tool support."""
    global _current_inference_tools
    _current_inference_tools = []  # Reset for this inference
    
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
                "description": "Search X (Twitter) for posts and discussions. ONLY use when the user EXPLICITLY asks to 'search X', 'check Twitter', 'what's trending', or 'what are people saying on X'. Do NOT use for general questions.",
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
                "description": "Search the web for information. ONLY use when the user EXPLICITLY asks to 'search the web', 'Google this', or 'look up'. Do NOT use for general knowledge questions you can answer directly.",
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
                "description": "Set a timer or reminder. Use this when the user asks to be reminded in X minutes/seconds, set a timer, or wants an alert after a duration.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "minutes": {
                            "type": "number",
                            "description": "Duration in minutes (can be decimal, e.g., 0.5 for 30 seconds)"
                        },
                        "seconds": {
                            "type": "number",
                            "description": "Duration in seconds (alternative to minutes, e.g., 10 for 10 seconds, 30 for 30 seconds)"
                        },
                        "message": {
                            "type": "string",
                            "description": "What to remind the user about (e.g., 'check the oven', 'take a break', 'meeting starts')"
                        }
                    },
                    "required": []
                }
            }
        },
        "list_timers": {
            "type": "function",
            "function": {
                "name": "list_timers",
                "description": "List all active timers and reminders. Use this when the user asks what timers are set, how much time is left, or wants to check their reminders.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        "set_reminder": {
            "type": "function",
            "function": {
                "name": "set_reminder",
                "description": "Set a reminder. Use this when the user asks to be reminded about something in X minutes. Same as set_timer but for reminder-style requests.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "minutes": {
                            "type": "number",
                            "description": "How many minutes from now (can be decimal, e.g., 0.5 for 30 seconds, 0.167 for 10 seconds)"
                        },
                        "seconds": {
                            "type": "number",
                            "description": "Alternative: seconds from now (e.g., 10 for 10 seconds, 30 for 30 seconds)"
                        },
                        "message": {
                            "type": "string",
                            "description": "What to remind the user about"
                        }
                    },
                    "required": ["message"]
                }
            }
        },
        "cancel_timer": {
            "type": "function",
            "function": {
                "name": "cancel_timer",
                "description": "Cancel an active timer or reminder. Use this when the user wants to cancel, stop, delete, or remove a timer or reminder.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Optional: the message/name of the timer to cancel. If not specified, cancels the most recent timer."
                        }
                    },
                    "required": []
                }
            }
        },
        "get_system_info": {
            "type": "function",
            "function": {
                "name": "get_system_info",
                "description": "Get information about the server/computer including CPU usage, memory usage, disk space, GPU utilization, and system uptime. Use this when the user asks about system status, server health, resource usage, or hardware information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "detail_level": {
                            "type": "string",
                            "enum": ["basic", "detailed"],
                            "description": "Level of detail: 'basic' for overview, 'detailed' to include top processes"
                        }
                    },
                    "required": []
                }
            }
        },
        "escalate_thinking": {
            "type": "function",
            "function": {
                "name": "escalate_thinking",
                "description": "Escalate complex reasoning to a more powerful cloud model for deep analysis. Use this when the user asks you to 'think deeply about', 'research', 'ponder', 'analyze thoroughly', or requests complex reasoning that would benefit from a superior model. The result will be available in future turns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question or topic to think deeply about"
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional additional context from the conversation to help with analysis"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    }
    
    # Filter tools based on persona permissions
    # Default all tools to enabled, then override with persona config
    default_permissions = {name: True for name in all_tools.keys()}
    merged_permissions = {**default_permissions, **(tool_permissions or {})}
    
    tools = None
    enabled_tools = [all_tools[name] for name, enabled in merged_permissions.items() if enabled and name in all_tools]
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
            # No tool calls - check if the model output JSON tool call in content
            content = message.get('content', '')
            parsed_result = try_parse_json_tool_call(content, auth)
            if parsed_result:
                return parsed_result
            return content
        
        tool_names = [tc['function']['name'] for tc in tool_calls]
        logger.info(f"Tool calls requested (iteration {iteration + 1}): {tool_names}")
        _current_inference_tools.extend(tool_names)
        
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


def try_parse_json_tool_call(content: str, auth: str) -> str:
    """Try to parse a JSON tool call from message content (fallback for models that output JSON)."""
    import re
    
    if not content:
        return None
    
    # Clean up content - remove trailing garbage after JSON
    content = content.strip()
    
    # Fix common JSON malformations
    # "parameters":} -> "parameters":{}}
    content = re.sub(r'"parameters"\s*:\s*\}', '"parameters":{}}', content)
    # "parameters": } -> "parameters":{}}
    content = re.sub(r'"parameters"\s*:\s*\}', '"parameters":{}}', content)
    # Remove trailing quotes/garbage
    content = re.sub(r'\}["\s]+$', '}', content)
    
    # Try to extract just the JSON object (handle trailing quotes/garbage)
    json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', content)
    if json_match:
        json_str = json_match.group(1)
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and 'name' in data:
                function_name = data['name']
                args = data.get('parameters', data.get('arguments', {}))
                
                logger.info(f"Parsed JSON tool call: {function_name}({args})")
                result = execute_tool(function_name, args, auth)
                return result
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse failed: {e}, trying regex patterns")
    
    # Look for JSON-like patterns with regex
    json_patterns = [
        r'["\']?name["\']?\s*:\s*["\'](\w+)["\'].*?["\']?parameters["\']?\s*:\s*(\{[^{}]*\})',
        r'["\']?function["\']?\s*:\s*["\'](\w+)["\'].*?["\']?arguments["\']?\s*:\s*(\{[^{}]*\})',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            function_name = match.group(1)
            try:
                # Try to parse the parameters
                params_str = match.group(2)
                # Clean up common issues
                params_str = params_str.replace("'", '"')
                args = json.loads(params_str)
                
                logger.info(f"Parsed JSON tool call from regex: {function_name}({args})")
                
                # Execute the tool
                result = execute_tool(function_name, args, auth)
                return result
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON tool params: {e}")
                continue
    
    # Last resort: just extract the tool name and call with empty args
    name_match = re.search(r'["\']?name["\']?\s*:\s*["\'](\w+)["\']', content, re.IGNORECASE)
    if name_match:
        function_name = name_match.group(1)
        logger.info(f"Extracted tool name only (no params): {function_name}")
        result = execute_tool(function_name, {}, auth)
        return result
    
    return None


def execute_tool(function_name: str, args: dict, auth: str) -> str:
    """Execute a tool and return the result."""
    
    # Normalize tool name aliases
    tool_aliases = {
        # Timer/reminder aliases
        'set_reminder': 'set_timer',
        'create_timer': 'set_timer',
        'create_reminder': 'set_timer',
        'add_timer': 'set_timer',
        'add_reminder': 'set_timer',
        'start_timer': 'set_timer',
        # List timer/reminder aliases
        'get_timers': 'list_timers',
        'get_reminders': 'list_timers',
        'list_reminders': 'list_timers',
        'get_timers_set': 'list_timers',
        'get_reminders_set': 'list_timers',
        'get_active_timers': 'list_timers',
        'get_active_reminders': 'list_timers',
        'show_timers': 'list_timers',
        'show_reminders': 'list_timers',
        'check_timers': 'list_timers',
        'check_reminders': 'list_timers',
        'active_timers': 'list_timers',
        'active_reminders': 'list_timers',
        # Cancel timer/reminder aliases
        'cancel_reminder': 'cancel_timer',
        'delete_timer': 'cancel_timer',
        'delete_reminder': 'cancel_timer',
        'remove_timer': 'cancel_timer',
        'remove_reminder': 'cancel_timer',
        'stop_timer': 'cancel_timer',
        'clear_timer': 'cancel_timer',
        'clear_reminder': 'cancel_timer',
        # DateTime aliases
        'get_time': 'get_current_datetime',
        'get_date': 'get_current_datetime',
        'current_time': 'get_current_datetime',
        'current_date': 'get_current_datetime',
        'what_time': 'get_current_datetime',
        # Weather aliases
        'get_weather': 'get_current_weather',
        'weather': 'get_current_weather',
        'check_weather': 'get_current_weather',
        # System info aliases
        'system_info': 'get_system_info',
        'system_status': 'get_system_info',
        'server_status': 'get_system_info',
        'server_info': 'get_system_info',
        'get_server_status': 'get_system_info',
        'get_server_info': 'get_system_info',
        'check_system': 'get_system_info',
        'hardware_info': 'get_system_info',
        'resource_usage': 'get_system_info',
        'get_gpu_info': 'get_system_info',
        'gpu_info': 'get_system_info',
        'get_gpu_status': 'get_system_info',
        'get_top_gpu_users': 'get_system_info',
        'gpu_processes': 'get_system_info',
        'get_gpu_processes': 'get_system_info',
        'get_cpu_info': 'get_system_info',
        'cpu_info': 'get_system_info',
        'get_memory_info': 'get_system_info',
        'memory_info': 'get_system_info',
        'get_disk_info': 'get_system_info',
        'disk_info': 'get_system_info',
        # Escalation aliases
        'think_deeply': 'escalate_thinking',
        'deep_think': 'escalate_thinking',
        'ponder': 'escalate_thinking',
        'analyze': 'escalate_thinking',
        'deep_analysis': 'escalate_thinking',
        'thorough_analysis': 'escalate_thinking',
        # Search aliases
        'search': 'search_x',
        'search_twitter': 'search_x',
        'twitter_search': 'search_x',
        'x_search': 'search_x',
        'search_internet': 'search_web',
        'internet_search': 'search_web',
        'google': 'search_web',
        'google_search': 'search_web',
        'web': 'search_web',
    }
    
    # Apply alias if exists
    function_name = tool_aliases.get(function_name, function_name)
    
    if function_name == 'get_current_datetime':
        return tool_get_current_datetime(args.get('timezone'))
    elif function_name == 'get_current_weather':
        return tool_get_current_weather(args.get('location', ''))
    elif function_name == 'set_timer':
        # Handle various parameter names the model might use
        minutes = args.get('minutes') or args.get('duration') or args.get('time')
        seconds = args.get('seconds')
        time_interval = args.get('time_interval')  # Model sometimes uses this
        
        # Parse time_interval if provided (e.g., "10 seconds", "5 minutes")
        if time_interval and not minutes and not seconds:
            minutes = parse_duration_string(str(time_interval))
        
        # If seconds is provided directly, convert to minutes
        if seconds is not None:
            try:
                minutes = float(seconds) / 60
            except (ValueError, TypeError):
                if isinstance(seconds, str):
                    minutes = parse_duration_string(seconds)
        elif minutes is not None:
            if isinstance(minutes, str):
                # Parse "1 minute", "5 minutes", "30 seconds" etc.
                minutes = parse_duration_string(minutes)
            else:
                minutes = float(minutes)
        else:
            minutes = 1.0  # Default
            
        message = args.get('message') or args.get('event') or args.get('reminder') or args.get('text') or ''
        logger.info(f"set_timer: minutes={minutes}, message={message}, raw_args={args}")
        return tool_set_timer(float(minutes), message)
    elif function_name == 'list_timers':
        return tool_list_timers()
    elif function_name == 'cancel_timer':
        return tool_cancel_timer(args.get('message') or args.get('timer') or args.get('which') or args.get('name'))
    elif function_name == 'get_system_info':
        return tool_get_system_info(args.get('detail_level', 'basic'))
    elif function_name == 'escalate_thinking':
        return tool_escalate_thinking(args.get('query', ''), args.get('context', ''), auth)
    elif function_name == 'search_x':
        return tool_search_x(args.get('query', ''), auth)
    elif function_name == 'search_web':
        return tool_search_web(args.get('query', ''), auth)
    else:
        # Fuzzy fallback - try to match unknown tools to known ones
        function_lower = function_name.lower()
        
        if 'timer' in function_lower or 'reminder' in function_lower:
            if 'list' in function_lower or 'get' in function_lower or 'show' in function_lower or 'check' in function_lower or 'active' in function_lower:
                logger.info(f"Fuzzy matched '{function_name}' to list_timers")
                return tool_list_timers()
            elif 'cancel' in function_lower or 'delete' in function_lower or 'remove' in function_lower or 'stop' in function_lower:
                logger.info(f"Fuzzy matched '{function_name}' to cancel_timer")
                return tool_cancel_timer(args.get('message') or args.get('timer') or args.get('which'))
            elif 'set' in function_lower or 'create' in function_lower or 'add' in function_lower or 'start' in function_lower:
                logger.info(f"Fuzzy matched '{function_name}' to set_timer")
                minutes = args.get('minutes') or args.get('duration') or args.get('time')
                seconds = args.get('seconds')
                if seconds:
                    minutes = float(seconds) / 60
                elif minutes and isinstance(minutes, str):
                    minutes = parse_duration_string(minutes)
                else:
                    minutes = float(minutes) if minutes else 1.0
                message = args.get('message') or args.get('event') or args.get('reminder') or ''
                return tool_set_timer(float(minutes), message)
        
        if 'time' in function_lower or 'date' in function_lower:
            logger.info(f"Fuzzy matched '{function_name}' to get_current_datetime")
            return tool_get_current_datetime(args.get('timezone'))
        
        if 'weather' in function_lower:
            logger.info(f"Fuzzy matched '{function_name}' to get_current_weather")
            return tool_get_current_weather(args.get('location', ''))
        
        if 'system' in function_lower or 'server' in function_lower or 'hardware' in function_lower or 'resource' in function_lower:
            logger.info(f"Fuzzy matched '{function_name}' to get_system_info")
            return tool_get_system_info(args.get('detail_level', 'basic'))
        
        if 'think' in function_lower or 'ponder' in function_lower or 'analyze' in function_lower or 'escalat' in function_lower:
            logger.info(f"Fuzzy matched '{function_name}' to escalate_thinking")
            return tool_escalate_thinking(args.get('query', ''), args.get('context', ''), auth)
        
        if 'search' in function_lower or 'google' in function_lower:
            query = args.get('query', '') or args.get('q', '') or args.get('term', '')
            if 'x' in function_lower or 'twitter' in function_lower:
                logger.info(f"Fuzzy matched '{function_name}' to search_x")
                return tool_search_x(query, auth)
            else:
                # Default to search_x for general search
                logger.info(f"Fuzzy matched '{function_name}' to search_x (default)")
                return tool_search_x(query, auth)
        
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


# ========== ESCALATE THINKING ==========
_thinking_rag_file = CONFIG_DIR / 'thinking_rag.json'
_escalation_rate_limit = {}  # {persona_id: [(timestamp, tokens_used), ...]}
_pending_thoughts = []  # List of pending/completed thoughts for announcement

def load_thinking_rag() -> dict:
    """Load the thinking RAG storage."""
    if _thinking_rag_file.exists():
        try:
            with open(_thinking_rag_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"insights": [], "usage": {"total_calls": 0, "total_tokens": 0}}

def save_thinking_rag(rag: dict):
    """Save the thinking RAG storage."""
    with open(_thinking_rag_file, 'w') as f:
        json.dump(rag, f, indent=2)

def check_escalation_rate_limit(persona_id: str, limit_per_hour: int = 10) -> tuple[bool, int]:
    """Check if escalation is allowed under rate limit. Returns (allowed, remaining)."""
    import time
    now = time.time()
    hour_ago = now - 3600
    
    # Clean old entries
    if persona_id in _escalation_rate_limit:
        _escalation_rate_limit[persona_id] = [
            (ts, tokens) for ts, tokens in _escalation_rate_limit[persona_id]
            if ts > hour_ago
        ]
    else:
        _escalation_rate_limit[persona_id] = []
    
    used = len(_escalation_rate_limit[persona_id])
    remaining = max(0, limit_per_hour - used)
    
    return remaining > 0, remaining

def extract_keywords(text: str) -> list:
    """Extract simple keywords from text for RAG retrieval."""
    import re
    # Remove common words and extract meaningful terms
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                 'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                 'from', 'as', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'between', 'under', 'again', 'further', 'then',
                 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
                 'and', 'but', 'if', 'or', 'because', 'until', 'while', 'about',
                 'against', 'this', 'that', 'these', 'those', 'what', 'which', 'who',
                 'whom', 'think', 'deeply', 'research', 'ponder', 'analyze', 'about',
                 'me', 'my', 'you', 'your', 'it', 'its', 'we', 'our', 'they', 'their'}
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = [w for w in words if w not in stopwords]
    # Return unique keywords, preserve order
    seen = set()
    return [w for w in keywords if not (w in seen or seen.add(w))][:10]

def background_escalate_thinking(query: str, context: str, persona_id: str, escalation_config: dict):
    """Background thread function to call cloud model and save result."""
    import time
    import uuid
    
    thought_id = str(uuid.uuid4())[:8]
    logger.info(f"[Escalation {thought_id}] Starting deep thinking: {query[:50]}...")
    
    try:
        # Build the reasoning prompt
        system_prompt = """You are a superior reasoning engine. Provide thorough, structured analysis.
Be comprehensive but concise. Use clear headings and bullet points where appropriate.
Focus on insights, implications, and actionable conclusions."""
        
        user_prompt = f"Analyze this deeply:\n\n{query}"
        if context:
            user_prompt += f"\n\nContext from conversation:\n{context}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the cloud model
        provider = escalation_config.get('provider', 'grok')
        url = escalation_config.get('url', 'https://api.x.ai/v1/chat/completions')
        auth = escalation_config.get('auth', '')
        model = escalation_config.get('model', 'grok-4-1-fast-reasoning')
        
        if not auth:
            # Try to get from environment
            auth = os.environ.get('XAI_API_KEY', '')
        
        if not auth:
            logger.error(f"[Escalation {thought_id}] No auth key for escalation")
            return
        
        headers = {
            'Authorization': f'Bearer {auth}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': model,
            'messages': messages,
            'temperature': 0.7,
            'max_tokens': 2000
        }
        
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        thinking_result = result['choices'][0]['message']['content']
        tokens_used = result.get('usage', {}).get('total_tokens', 0)
        elapsed = time.time() - start_time
        
        logger.info(f"[Escalation {thought_id}] Complete in {elapsed:.1f}s, {tokens_used} tokens")
        
        # Extract keywords for retrieval
        keywords = extract_keywords(query + " " + thinking_result)
        
        # Save to RAG
        rag = load_thinking_rag()
        insight = {
            "id": thought_id,
            "query": query,
            "context": context[:500] if context else "",  # Limit stored context
            "response": thinking_result,
            "keywords": keywords,
            "timestamp": time.time(),
            "persona_id": persona_id,
            "model": model,
            "tokens_used": tokens_used
        }
        rag["insights"].append(insight)
        rag["usage"]["total_calls"] += 1
        rag["usage"]["total_tokens"] += tokens_used
        save_thinking_rag(rag)
        
        # Track rate limit
        if persona_id not in _escalation_rate_limit:
            _escalation_rate_limit[persona_id] = []
        _escalation_rate_limit[persona_id].append((time.time(), tokens_used))
        
        # Add to pending thoughts for announcement
        _pending_thoughts.append({
            "id": thought_id,
            "query": query,
            "response": thinking_result,
            "persona_id": persona_id,
            "timestamp": time.time(),
            "announced": False
        })
        
        logger.info(f"[Escalation {thought_id}] Saved to RAG with {len(keywords)} keywords")
        
    except Exception as e:
        logger.error(f"[Escalation {thought_id}] Failed: {e}")

def retrieve_relevant_insights(query: str, max_results: int = 3) -> list:
    """Retrieve relevant insights from RAG store based on keyword matching."""
    rag = load_thinking_rag()
    insights = rag.get('insights', [])
    
    if not insights:
        return []
    
    query_keywords = set(extract_keywords(query))
    
    if not query_keywords:
        return []
    
    # Score insights by keyword overlap
    scored = []
    for insight in insights:
        insight_keywords = set(insight.get('keywords', []))
        overlap = len(query_keywords & insight_keywords)
        if overlap > 0:
            scored.append((overlap, insight))
    
    # Sort by score and return top results
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:max_results]]


def format_insights_for_context(insights: list) -> str:
    """Format retrieved insights as context for the model."""
    if not insights:
        return ""
    
    lines = ["\n[Previous Deep Analysis - Use these insights if relevant to the user's question:]"]
    for i, insight in enumerate(insights, 1):
        query = insight.get('query', '')[:100]
        response = insight.get('response', '')[:500]  # Limit to avoid context overflow
        lines.append(f"\n--- Insight {i}: Re: \"{query}\" ---")
        lines.append(response)
        if len(insight.get('response', '')) > 500:
            lines.append("... (truncated)")
    
    return "\n".join(lines)


def tool_escalate_thinking(query: str, context: str = '', auth: str = '') -> str:
    """Escalate complex reasoning to a cloud model. Returns immediately, processes in background."""
    import time
    
    if not query:
        return "I need a question or topic to think deeply about."
    
    # Get active persona and escalation config
    config = load_config()
    persona_id = config.get('active_persona', 'default')
    persona = config.get('personas', {}).get(persona_id, {})
    escalation_config = persona.get('escalation', {})
    
    if not escalation_config.get('enabled', True):
        return "Deep thinking is not enabled for this persona."
    
    # Check rate limit
    limit = escalation_config.get('rate_limit_per_hour', 10)
    allowed, remaining = check_escalation_rate_limit(persona_id, limit)
    
    if not allowed:
        return f"I've reached my deep thinking limit for now. Please try again in about an hour."
    
    # Spawn background thread
    thread = threading.Thread(
        target=background_escalate_thinking,
        args=(query, context, persona_id, escalation_config),
        daemon=True
    )
    thread.start()
    
    # Return immediate acknowledgment
    ack_phrases = [
        f"Let me ponder that deeply... I'll have insights ready shortly. ({remaining-1} deep thoughts remaining this hour)",
        f"Researching that thoroughly in the background... ({remaining-1} remaining)",
        f"Thinking deeply about that... I'll integrate my analysis into our conversation. ({remaining-1} remaining)",
    ]
    import random
    return random.choice(ack_phrases)


def tool_get_system_info(detail_level: str = 'basic') -> str:
    """Get system information including CPU, memory, disk, and GPU usage."""
    import psutil
    import platform
    import subprocess
    
    info = {}
    
    # Basic system info
    info['system'] = f"{platform.system()} {platform.release()}"
    info['hostname'] = platform.node()
    info['architecture'] = platform.machine()
    
    # CPU model name - try lscpu first, then /proc/cpuinfo
    cpu_model = None
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            models = []
            for line in result.stdout.split('\n'):
                if 'Model name:' in line:
                    model = line.split(':')[1].strip()
                    if model not in models:
                        models.append(model)
            if models:
                cpu_model = ' / '.join(models)
    except:
        pass
    
    if not cpu_model:
        cpu_model = platform.processor()
        if not cpu_model or cpu_model == 'aarch64':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line.lower():
                            cpu_model = line.split(':')[1].strip()
                            break
            except:
                pass
    info['cpu_model'] = cpu_model or 'Unknown'
    
    # Product name (useful for branded systems like DGX)
    try:
        with open('/sys/devices/virtual/dmi/id/product_name', 'r') as f:
            info['product_name'] = f.read().strip().replace('_', ' ')
    except:
        info['product_name'] = None
    
    # CPU info
    info['cpu_percent'] = psutil.cpu_percent(interval=0.5)
    info['cpu_cores'] = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        info['cpu_freq_mhz'] = int(cpu_freq.current)
    
    # Memory info
    mem = psutil.virtual_memory()
    info['memory_percent'] = mem.percent
    info['memory_used_gb'] = round(mem.used / (1024**3), 1)
    info['memory_total_gb'] = round(mem.total / (1024**3), 1)
    
    # Disk info
    disk = psutil.disk_usage('/')
    info['disk_percent'] = disk.percent
    info['disk_used_gb'] = int(disk.used / (1024**3))
    info['disk_total_gb'] = int(disk.total / (1024**3))
    info['disk_free_gb'] = int(disk.free / (1024**3))
    
    # GPU info (try nvidia-smi)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    name, mem_used, mem_total, util, temp = parts[:5]
                    gpus.append({
                        'name': name,
                        'utilization_percent': util,
                        'memory_used_mb': mem_used if mem_used != '[N/A]' else 'unknown',
                        'memory_total_mb': mem_total if mem_total != '[N/A]' else 'unknown',
                        'temperature_c': temp
                    })
            info['gpus'] = gpus
        
        # Get GPU processes
        proc_result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,name,used_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if proc_result.returncode == 0 and proc_result.stdout.strip():
            gpu_processes = []
            for line in proc_result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        pid, name, mem = parts[:3]
                        gpu_processes.append({
                            'pid': pid,
                            'name': name,
                            'memory_mb': mem
                        })
            if gpu_processes:
                info['gpu_processes'] = gpu_processes
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"nvidia-smi not available: {e}")
    
    # Uptime
    boot_time = psutil.boot_time()
    import time
    uptime_seconds = time.time() - boot_time
    info['uptime_days'] = int(uptime_seconds // 86400)
    info['uptime_hours'] = int((uptime_seconds % 86400) // 3600)
    info['uptime_minutes'] = int((uptime_seconds % 3600) // 60)
    
    # Format as readable text (no emojis for TTS)
    system_line = f"System: {info['system']} on {info['hostname']}, {info['architecture']} architecture"
    if info.get('product_name'):
        system_line = f"System: {info['product_name']}, {info['system']}, {info['architecture']} architecture"
    
    lines = [
        system_line,
        f"CPU: {info['cpu_model']}, {info['cpu_cores']} cores, {info['cpu_percent']}% used",
        f"Memory: {info['memory_percent']}% used, {info['memory_used_gb']} of {info['memory_total_gb']} GB",
        f"Disk: {info['disk_percent']}% used, {info['disk_free_gb']} GB free of {info['disk_total_gb']} GB total",
    ]
    
    if 'gpus' in info:
        for i, gpu in enumerate(info['gpus']):
            mem_info = ""
            if gpu['memory_used_mb'] != 'unknown' and gpu['memory_total_mb'] != 'unknown':
                mem_info = f", {gpu['memory_used_mb']} of {gpu['memory_total_mb']} MB VRAM used"
            lines.append(f"GPU {i}: {gpu['name']}, {gpu['utilization_percent']}% utilized{mem_info}, temperature {gpu['temperature_c']} degrees Celsius")
    
    if 'gpu_processes' in info:
        lines.append(f"GPU processes ({len(info['gpu_processes'])} running):")
        for proc in info['gpu_processes'][:5]:  # Limit to top 5
            lines.append(f"  - {proc['name']} (PID {proc['pid']}): {proc['memory_mb']} MB")
    
    lines.append(f"Uptime: {info['uptime_days']} days, {info['uptime_hours']} hours, {info['uptime_minutes']} minutes")
    
    return "\n".join(lines)


def parse_duration_string(duration_str: str) -> float:
    """Parse duration strings like '5 minutes', '30 seconds', '1 hour' into minutes."""
    import re
    
    if not duration_str:
        return 1.0
        
    duration_str = str(duration_str).lower().strip()
    
    # Try to extract number and unit (handle plurals)
    match = re.search(r'(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)?', duration_str)
    if match:
        value = float(match.group(1))
        unit = match.group(2) or 'minute'
        
        logger.debug(f"Parsed duration: {value} {unit}")
        
        if unit.startswith('second') or unit.startswith('sec') or unit == 's':
            return value / 60
        elif unit.startswith('hour') or unit.startswith('hr') or unit == 'h':
            return value * 60
        else:  # minutes
            return value
    
    # Try to parse as just a number (assume minutes)
    try:
        return float(duration_str)
    except ValueError:
        pass
    
    # Default to 1 minute if can't parse
    logger.warning(f"Could not parse duration: {duration_str}, defaulting to 1 minute")
    return 1.0


def tool_list_timers() -> str:
    """List all active timers."""
    import time
    
    current_time = time.time()
    
    with _timer_lock:
        active = [t for t in _active_timers if t['status'] == 'active']
    
    if not active:
        return "No active timers or reminders."
    
    lines = [f"You have {len(active)} active timer{'s' if len(active) != 1 else ''}:"]
    
    for timer in active:
        remaining = max(0, timer['expires_at'] - current_time)
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        
        if mins > 0:
            time_left = f"{mins} minute{'s' if mins != 1 else ''} and {secs} second{'s' if secs != 1 else ''}"
        else:
            time_left = f"{secs} second{'s' if secs != 1 else ''}"
        
        message = timer.get('message', 'Timer') or 'Timer'
        lines.append(f"- {message}: {time_left} remaining")
    
    return "\n".join(lines)


def tool_cancel_timer(message_or_index: str = None) -> str:
    """Cancel a timer by message text or index (most recent first)."""
    import time
    
    with _timer_lock:
        active = [t for t in _active_timers if t['status'] == 'active']
        
        if not active:
            return "No active timers to cancel."
        
        # Sort by creation time (most recent first for index-based cancellation)
        active.sort(key=lambda t: t.get('created_at', 0), reverse=True)
        
        # Try to find by message match
        if message_or_index:
            message_lower = str(message_or_index).lower().strip()
            
            # Try to parse as index (1-based)
            try:
                idx = int(message_lower) - 1
                if 0 <= idx < len(active):
                    timer = active[idx]
                    timer['status'] = 'cancelled'
                    return f"Cancelled timer: {timer.get('message', 'Timer')}"
            except ValueError:
                pass
            
            # Find by message content
            for timer in active:
                timer_msg = (timer.get('message', '') or '').lower()
                if message_lower in timer_msg or timer_msg in message_lower:
                    timer['status'] = 'cancelled'
                    return f"Cancelled timer: {timer.get('message', 'Timer')}"
            
            # If specific message not found, cancel the most recent
            timer = active[0]
            timer['status'] = 'cancelled'
            return f"Could not find timer matching '{message_or_index}', cancelled most recent: {timer.get('message', 'Timer')}"
        
        # No message specified, cancel the most recent
        timer = active[0]
        timer['status'] = 'cancelled'
        return f"Cancelled timer: {timer.get('message', 'Timer')}"


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


# ========== ESCALATION ENDPOINTS ==========

@app.route('/api/escalation/status', methods=['GET'])
def get_escalation_status():
    """Get escalation rate limit status and usage stats."""
    config = load_config()
    persona_id = config.get('active_persona', 'default')
    persona = config.get('personas', {}).get(persona_id, {})
    escalation_config = persona.get('escalation', {})
    
    limit = escalation_config.get('rate_limit_per_hour', 10)
    allowed, remaining = check_escalation_rate_limit(persona_id, limit)
    
    rag = load_thinking_rag()
    
    return jsonify({
        'success': True,
        'enabled': escalation_config.get('enabled', True),
        'rate_limit': {
            'per_hour': limit,
            'remaining': remaining,
            'used': limit - remaining
        },
        'usage': rag.get('usage', {}),
        'total_insights': len(rag.get('insights', []))
    })


@app.route('/api/escalation/pending', methods=['GET'])
def get_pending_thoughts():
    """Get any pending thoughts that are ready to announce."""
    config = load_config()
    persona_id = config.get('active_persona', 'default')
    
    ready = []
    for thought in _pending_thoughts:
        if thought['persona_id'] == persona_id and not thought['announced']:
            ready.append({
                'id': thought['id'],
                'query': thought['query'],
                'response': thought['response'],
                'timestamp': thought['timestamp']
            })
    
    return jsonify({
        'success': True,
        'pending': ready
    })


@app.route('/api/escalation/pending/<thought_id>/acknowledge', methods=['POST'])
def ack_pending_thought(thought_id):
    """Mark a pending thought as announced."""
    for thought in _pending_thoughts:
        if thought['id'] == thought_id:
            thought['announced'] = True
            logger.info(f"Acknowledged thought {thought_id}")
            break
    return jsonify({'success': True})


@app.route('/api/escalation/insights', methods=['GET'])
def get_insights():
    """Get recent insights from the RAG store."""
    rag = load_thinking_rag()
    insights = rag.get('insights', [])
    
    # Return last 10, most recent first
    recent = sorted(insights, key=lambda x: x.get('timestamp', 0), reverse=True)[:10]
    
    return jsonify({
        'success': True,
        'insights': recent
    })


def tool_search_x(query: str, auth: str) -> str:
    """Search X (Twitter) using the xAI SDK."""
    if not query:
        return "No search query provided"
    
    logger.info(f"Searching X for: {query}")
    
    # Get xAI API key - prefer explicit auth, then escalation config, then env var
    xai_key = auth
    if not xai_key:
        config = load_config()
        persona_id = config.get('active_persona', 'default')
        persona = config.get('personas', {}).get(persona_id, {})
        xai_key = persona.get('escalation', {}).get('auth', '')
    if not xai_key:
        xai_key = os.environ.get('XAI_API_KEY', '')
    
    if not xai_key:
        return "X search requires xAI API key. Set XAI_API_KEY or configure escalation auth."
    
    try:
        from xai_sdk import Client
        from xai_sdk.chat import user
        from xai_sdk.tools import x_search
        
        client = Client(api_key=xai_key)
        
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
    
    # Get xAI API key - prefer explicit auth, then escalation config, then env var
    xai_key = auth
    if not xai_key:
        config = load_config()
        persona_id = config.get('active_persona', 'default')
        persona = config.get('personas', {}).get(persona_id, {})
        xai_key = persona.get('escalation', {}).get('auth', '')
    if not xai_key:
        xai_key = os.environ.get('XAI_API_KEY', '')
    
    if not xai_key:
        return "Web search requires xAI API key. Set XAI_API_KEY or configure escalation auth."
    
    try:
        from xai_sdk import Client
        from xai_sdk.chat import user
        from xai_sdk.tools import web_search
        
        client = Client(api_key=xai_key)
        
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

