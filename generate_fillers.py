#!/usr/bin/env python3
"""
Generate persona-specific filler sounds using Chatterbox TTS.
These are short utterances that make the AI sound more natural while processing.
"""

import os
import sys
import json
from pathlib import Path

# Filler categories and phrases
FILLERS = {
    # Short hesitation sounds (vary the spelling for different vocalizations)
    "hesitation": [
        ("uh.wav", "uh"),
        ("uhh.wav", "uhh"),
        ("uhhh.wav", "uhhh..."),
        ("uh-huh.wav", "uh huh"),
        ("um.wav", "um"),
        ("umm.wav", "umm"),
        ("ummm.wav", "ummm..."),
        ("uhm.wav", "uhm"),
        ("mmm.wav", "mmm"),
        ("hmm.wav", "hmm"),
        ("hmmm.wav", "hmmm..."),
    ],
    
    # Short acknowledgment sounds
    "acknowledgment": [
        ("ah.wav", "ah"),
        ("oh.wav", "oh"),
        ("ooh.wav", "ooh"),
        ("so.wav", "so"),
    ],
    
    # Thinking phrases
    "thinking": [
        ("let-me-think.wav", "let me think"),
        ("let-me-see.wav", "let me see"),
        ("give-me-a-moment.wav", "give me a moment"),
        ("one-moment.wav", "one moment"),
        ("well.wav", "well"),
        ("well....wav", "well..."),
    ],
    
    # Tool call affirmations (start with these before tool calls)
    "tool_affirmation": [
        ("ok.wav", "ok"),
        ("ok....wav", "ok..."),
        ("okay.wav", "okay"),
        ("yes.wav", "yes"),
        ("yes....wav", "yes..."),
        ("sure.wav", "sure"),
        ("sure....wav", "sure..."),
        ("alright.wav", "alright"),
        ("alright....wav", "alright..."),
    ],
    
    # Tool call transitions (after affirmation)
    "tool_transition": [
        ("hang-on.wav", "hang on"),
        ("hang-on....wav", "hang on..."),
        ("just-a-second.wav", "just a second"),
        ("just-a-moment.wav", "just a moment"),
        ("one-sec.wav", "one sec"),
        ("checking.wav", "checking"),
        ("checking....wav", "checking..."),
        ("looking-it-up.wav", "looking it up"),
        ("let-me-check.wav", "let me check"),
        ("let-me-look.wav", "let me look"),
    ],
}


def get_persona_voice(persona_name: str) -> str:
    """Get the voice file path for a persona from config."""
    config_path = Path("config/api_settings.json")
    if not config_path.exists():
        return None
    
    with open(config_path) as f:
        config = json.load(f)
    
    personas = config.get("personas", {})
    persona = personas.get(persona_name, {})
    
    synthesis = persona.get("synthesis", {})
    voice = synthesis.get("voice", "")
    
    if voice and Path(f"static/voices/{voice}").exists():
        return f"static/voices/{voice}"
    
    return None


def generate_filler_with_chatterbox(text: str, output_path: str, voice_path: str):
    """Generate a filler using Chatterbox TTS."""
    try:
        import torch
        from chatterbox.tts_turbo import ChatterboxTurbo
        import numpy as np
        from scipy.io import wavfile
        
        # Initialize Chatterbox
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTurbo.from_pretrained(device=device)
        
        # Generate audio
        print(f"  Generating: '{text}' -> {output_path}")
        wav = model.generate(text, audio_prompt_path=voice_path)
        
        # Save as WAV
        wav_np = wav.squeeze().cpu().numpy()
        wav_int16 = (wav_np * 32767).astype(np.int16)
        wavfile.write(output_path, 24000, wav_int16)
        
        return True
        
    except Exception as e:
        print(f"  Error generating '{text}': {e}")
        return False


def generate_fillers_for_persona(persona_name: str, categories: list = None):
    """Generate all fillers for a persona."""
    voice_path = get_persona_voice(persona_name)
    if not voice_path:
        print(f"No voice found for persona '{persona_name}'")
        return
    
    print(f"Using voice: {voice_path}")
    
    output_dir = Path(f"static/fillers/{persona_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter categories if specified
    if categories:
        fillers_to_generate = {k: v for k, v in FILLERS.items() if k in categories}
    else:
        fillers_to_generate = FILLERS
    
    # Load model once
    try:
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        import numpy as np
        from scipy.io import wavfile
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Chatterbox on {device}...")
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        
    except ImportError as e:
        print(f"Chatterbox not available: {e}")
        return
    
    generated = 0
    skipped = 0
    
    for category, phrases in fillers_to_generate.items():
        print(f"\n=== {category.upper()} ===")
        
        for filename, text in phrases:
            output_path = output_dir / filename
            
            # Skip if already exists
            if output_path.exists():
                print(f"  Skipping (exists): {filename}")
                skipped += 1
                continue
            
            try:
                print(f"  Generating: '{text}' -> {filename}")
                wav = model.generate(text, audio_prompt_path=voice_path)
                
                wav_np = wav.squeeze().cpu().numpy()
                wav_int16 = (wav_np * 32767).astype(np.int16)
                wavfile.write(str(output_path), 24000, wav_int16)
                
                generated += 1
                
            except Exception as e:
                print(f"  Error: {e}")
    
    print(f"\n✓ Generated {generated} new fillers, skipped {skipped} existing")
    print(f"  Output: {output_dir}/")


def list_categories():
    """List available filler categories."""
    print("Available filler categories:")
    for category, phrases in FILLERS.items():
        print(f"  {category}: {len(phrases)} phrases")
        for filename, text in phrases[:3]:
            print(f"    - {filename}: \"{text}\"")
        if len(phrases) > 3:
            print(f"    ... and {len(phrases) - 3} more")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate persona fillers with Chatterbox TTS")
    parser.add_argument("persona", nargs="?", help="Persona name (e.g., local_assistant_2590)")
    parser.add_argument("--categories", "-c", nargs="+", choices=list(FILLERS.keys()),
                        help="Only generate specific categories")
    parser.add_argument("--list", "-l", action="store_true", help="List available categories")
    parser.add_argument("--force", "-f", action="store_true", help="Regenerate existing files")
    
    args = parser.parse_args()
    
    if args.list:
        list_categories()
        return
    
    if not args.persona:
        parser.print_help()
        print("\nAvailable personas:")
        config_path = Path("config/api_settings.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            for name in config.get("personas", {}).keys():
                print(f"  - {name}")
        return
    
    generate_fillers_for_persona(args.persona, args.categories)


if __name__ == "__main__":
    main()
