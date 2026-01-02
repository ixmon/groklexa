#!/usr/bin/env python3
"""
Generate stock earcon sounds using synthesized tones.
These are fallback sounds when persona-specific fillers aren't available.
"""

import numpy as np
from scipy.io import wavfile
import os

SAMPLE_RATE = 24000
OUTPUT_DIR = "static/earcons"

def generate_tone(frequency, duration, volume=0.3, fade_in=0.02, fade_out=0.05):
    """Generate a sine wave tone with fade in/out."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = np.sin(2 * np.pi * frequency * t) * volume
    
    # Apply fade in
    fade_in_samples = int(SAMPLE_RATE * fade_in)
    tone[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
    
    # Apply fade out
    fade_out_samples = int(SAMPLE_RATE * fade_out)
    tone[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)
    
    return tone

def generate_ding():
    """Pleasant single ding for acknowledgment."""
    # Two harmonics for a richer sound
    t = np.linspace(0, 0.4, int(SAMPLE_RATE * 0.4), False)
    tone = (np.sin(2 * np.pi * 880 * t) * 0.3 + 
            np.sin(2 * np.pi * 1760 * t) * 0.1)
    
    # Exponential decay
    decay = np.exp(-t * 8)
    tone *= decay
    
    return tone

def generate_thinking():
    """Soft processing sound - rising tone."""
    duration = 0.5
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    # Frequency sweep from 300 to 500 Hz
    freq = 300 + 200 * (t / duration)
    tone = np.sin(2 * np.pi * freq * t) * 0.2
    
    # Fade out
    fade_samples = int(SAMPLE_RATE * 0.15)
    tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    return tone

def generate_active():
    """Double beep for activation."""
    beep1 = generate_tone(523, 0.1, volume=0.25)  # C5
    silence = np.zeros(int(SAMPLE_RATE * 0.08))
    beep2 = generate_tone(659, 0.1, volume=0.25)  # E5
    
    return np.concatenate([beep1, silence, beep2])

def generate_speaking():
    """Soft tone indicating AI is about to speak."""
    duration = 0.3
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    # Chord: C4 + E4 + G4
    tone = (np.sin(2 * np.pi * 262 * t) * 0.15 +
            np.sin(2 * np.pi * 330 * t) * 0.1 +
            np.sin(2 * np.pi * 392 * t) * 0.08)
    
    # Fade in and out
    fade_samples = int(SAMPLE_RATE * 0.1)
    tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
    tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    return tone

def generate_tool_calling():
    """Processing/working sound - mechanical feel."""
    duration = 0.6
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    # Two alternating tones
    freq = 440 + 100 * np.sin(2 * np.pi * 8 * t)  # Wobble effect
    tone = np.sin(2 * np.pi * freq * t) * 0.2
    
    # Add slight noise for texture
    noise = np.random.randn(len(t)) * 0.02
    tone = tone + noise
    
    # Fade out
    fade_samples = int(SAMPLE_RATE * 0.2)
    tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    return tone

def generate_error():
    """Descending tone for errors."""
    duration = 0.4
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    # Frequency sweep down
    freq = 500 - 200 * (t / duration)
    tone = np.sin(2 * np.pi * freq * t) * 0.25
    
    # Fade out
    fade_samples = int(SAMPLE_RATE * 0.1)
    tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    return tone

def generate_waiting():
    """Subtle ambient tone for longer waits."""
    duration = 1.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    # Very low, subtle drone
    tone = (np.sin(2 * np.pi * 220 * t) * 0.08 +
            np.sin(2 * np.pi * 330 * t) * 0.05)
    
    # Slow pulse
    pulse = 0.8 + 0.2 * np.sin(2 * np.pi * 1.5 * t)
    tone *= pulse
    
    # Fade in and out
    fade_samples = int(SAMPLE_RATE * 0.2)
    tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
    tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    return tone

def save_wav(filename, audio):
    """Save audio as 16-bit WAV file."""
    # Normalize and convert to int16
    audio = np.clip(audio, -1, 1)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    wavfile.write(filepath, SAMPLE_RATE, audio_int16)
    print(f"Generated: {filepath}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    earcons = {
        "ding.wav": generate_ding(),
        "thinking.wav": generate_thinking(),
        "active.wav": generate_active(),
        "speaking.wav": generate_speaking(),
        "tool_calling.wav": generate_tool_calling(),
        "error.wav": generate_error(),
        "waiting.wav": generate_waiting(),
    }
    
    for filename, audio in earcons.items():
        save_wav(filename, audio)
    
    print(f"\nGenerated {len(earcons)} earcon files in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()


