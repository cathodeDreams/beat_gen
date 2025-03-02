import numpy as np
from scipy.io import wavfile
from scipy import signal

# Import functions from existing scripts
from beat_gen.edm_kick import generate_edm_kick
from beat_gen.edm_snare import generate_edm_snare, layered_snare
from beat_gen.glitch_percussion_generator import generate_glitch_percussion
from beat_gen.sub_bass_generator import (generate_sub_bass, create_wobble_bass, 
                                         create_808_bass, create_reese_bass)

class DrumSound:
    """Class representing a drum sound with metadata and audio data"""
    def __init__(self, name, category, audio_data, sample_rate=44100):
        self.name = name
        self.category = category  # kick, snare, hat, percussion, etc.
        self.audio_data = audio_data
        self.sample_rate = sample_rate
    
    def save(self, filename=None):
        """Save the sound as a WAV file"""
        if filename is None:
            filename = f"{self.name.lower().replace(' ', '_')}.wav"
        
        wavfile.write(filename, self.sample_rate, (self.audio_data * 32767).astype(np.int16))
        return filename

# Factory functions for creating drum sounds

def create_kick(style="standard", **kwargs):
    """Create a kick drum sound"""
    styles = {
        "standard": {"freq_start": 180, "freq_end": 40, "pitch_bend_speed": 18, 
                    "punch_amount": 1.2, "distortion": 0.3},
        "deep_house": {"freq_start": 140, "freq_end": 35, "pitch_bend_speed": 12, 
                      "punch_amount": 0.8, "distortion": 0.2, "texture_amount": 0.1, 
                      "body_resonance": 0.5},
        "hardstyle": {"freq_start": 200, "freq_end": 45, "pitch_bend_speed": 25, 
                     "punch_amount": 1.5, "distortion": 0.7, "texture_amount": 0.4, 
                      "body_resonance": 0.7},
        "vump": {"freq_start": 160, "freq_end": 30, "pitch_bend_speed": 20, 
                "punch_amount": 1.8, "distortion": 0.5, "texture_amount": 0.3, 
                "body_resonance": 0.6},
        "techno": {"freq_start": 150, "freq_end": 45, "pitch_bend_speed": 15, 
                  "punch_amount": 1.0, "distortion": 0.25, "texture_amount": 0.2}
    }
    
    # Get base parameters from style
    params = styles.get(style, styles["standard"]).copy()
    
    # Override with any provided kwargs
    params.update(kwargs)
    
    # Generate kick audio
    audio_data = generate_edm_kick(**params)
    
    return DrumSound(f"{style.capitalize()} Kick", "kick", audio_data)

def create_snare(style="standard", **kwargs):
    """Create a snare drum sound"""
    styles = {
        "standard": {"body_freq": 180, "noise_amount": 0.8, "snap_amount": 0.6, 
                    "body_amount": 0.4, "distortion": 0.2, "reverb_amount": 0.2},
        "trap": {"body_freq": 220, "noise_amount": 0.7, "snap_amount": 0.9, 
                "body_amount": 0.3, "body_decay": 20, "noise_decay": 12, 
                "distortion": 0.3, "bandpass_high": 8000, "reverb_amount": 0.15},
        "dubstep": {"body_freq": 150, "noise_amount": 0.9, "snap_amount": 0.8, 
                   "body_amount": 0.6, "body_decay": 10, "noise_decay": 7, 
                   "distortion": 0.5, "bandpass_low": 300, "bandpass_high": 7000, 
                   "reverb_amount": 0.3},
        "future_bass": {"body_freq": 200, "noise_amount": 0.85, "snap_amount": 0.7, 
                       "body_amount": 0.35, "body_decay": 12, "noise_decay": 6, 
                       "distortion": 0.1, "bandpass_low": 500, "bandpass_high": 10000, 
                       "reverb_amount": 0.4},
        "hardstyle": {"body_freq": 180, "noise_amount": 0.95, "snap_amount": 0.9, 
                     "body_amount": 0.5, "body_decay": 8, "noise_decay": 5, 
                     "distortion": 0.7, "bandpass_low": 300, "bandpass_high": 9000, 
                     "reverb_amount": 0.25}
    }
    
    # Get base parameters from style
    params = styles.get(style, styles["standard"]).copy()
    
    # Override with any provided kwargs
    params.update(kwargs)
    
    # Special case for layered snare
    if style == "layered":
        audio_data = layered_snare()
    else:
        # Generate snare audio
        audio_data = generate_edm_snare(**params)
    
    return DrumSound(f"{style.capitalize()} Snare", "snare", audio_data)

def create_hat(style="closed", **kwargs):
    """Create a hi-hat sound using filtered noise"""
    # Define sample rate
    sample_rate = kwargs.get("sample_rate", 44100)
    
    # Default durations for different hat types
    durations = {
        "closed": 0.1,
        "open": 0.3,
        "pedal": 0.15
    }
    
    # Get duration from style or override with kwargs
    duration = kwargs.get("duration", durations.get(style, 0.1))
    
    # Generate white noise
    num_samples = int(sample_rate * duration)
    noise = np.random.uniform(-1, 1, num_samples)
    
    # Define different frequency ranges for different hat styles
    freqs = {
        "closed": (4000, 16000),
        "open": (2000, 14000),
        "pedal": (800, 8000)
    }
    
    # Get frequency range from style or override with kwargs
    freq_range = kwargs.get("freq_range", freqs.get(style, (3000, 12000)))
    
    # Apply bandpass filter
    sos = signal.butter(4, freq_range, 'bp', fs=sample_rate, output='sos')
    filtered = signal.sosfilt(sos, noise)
    
    # Define different decay rates for different hat styles
    decays = {
        "closed": 50,
        "open": 10,
        "pedal": 30
    }
    
    # Get decay rate from style or override with kwargs
    decay = kwargs.get("decay", decays.get(style, 40))
    
    # Apply envelope
    t = np.linspace(0, duration, num_samples)
    env = np.exp(-t * decay)
    
    # Add resonance
    if kwargs.get("resonance", True):
        resonance_freq = kwargs.get("resonance_freq", 8000)
        q = kwargs.get("q", 10)
        b, a = signal.iirpeak(resonance_freq, q, sample_rate)
        filtered = signal.lfilter(b, a, filtered)
    
    # Combine envelope and audio
    audio_data = filtered * env * kwargs.get("volume", 0.8)
    
    return DrumSound(f"{style.capitalize()} Hat", "hat", audio_data)

def create_percussion(style="click", **kwargs):
    """Create a percussion sound using the glitch percussion generator"""
    # Generate percussion audio
    audio_data = generate_glitch_percussion(style=style, **kwargs)
    
    return DrumSound(f"{style.capitalize()} Percussion", "percussion", audio_data)

def create_sub_bass(style="basic", **kwargs):
    """Create a sub-bass sound"""
    styles = {
        "basic": {"waveform": "sine", "duration": 1.0, "base_freq": 40},
        "808": {"format_type": "trap", "duration": 1.0, "base_freq": 40},
        "wobble": {"format_type": "dubstep", "duration": 1.0, "base_freq": 40, "wobble_rate": 2.0},
        "reese": {"duration": 1.0, "base_freq": 40, "num_oscillators": 3}
    }
    
    # Get base parameters from style
    params = styles.get(style, styles["basic"]).copy()
    
    # Override with any provided kwargs
    params.update(kwargs)
    
    # Generate sub-bass audio based on style
    if style == "808":
        audio_data = create_808_bass(**params)
    elif style == "wobble":
        audio_data = create_wobble_bass(**params)
    elif style == "reese":
        audio_data = create_reese_bass(**params)
    else:  # basic
        audio_data = generate_sub_bass(**params)
    
    return DrumSound(f"{style.capitalize()} Sub", "sub_bass", audio_data)

# Drum Kit Class to hold a collection of sounds
class DrumKit:
    """Class representing a complete drum kit"""
    
    def __init__(self, name="EDM Kit"):
        self.name = name
        self.sounds = {}
    
    def add_sound(self, sound):
        """Add a sound to the kit"""
        self.sounds[sound.name] = sound
        return self
    
    def get_sound(self, name):
        """Get a sound by name"""
        return self.sounds.get(name)
    
    def get_sounds_by_category(self, category):
        """Get all sounds of a particular category"""
        return [sound for sound in self.sounds.values() if sound.category == category]
    
    def get_sub_bass(self):
        """Get the first sub-bass sound in the kit"""
        sub_basses = self.get_sounds_by_category("sub_bass")
        return sub_basses[0] if sub_basses else None
    
    def save_kit(self, folder="."):
        """Save all sounds in the kit to WAV files"""
        for sound in self.sounds.values():
            sound.save(f"{folder}/{self.name.lower().replace(' ', '_')}_{sound.name.lower().replace(' ', '_')}.wav")

# Create preset kits for different EDM styles

def create_house_kit():
    """Create a house music drum kit"""
    kit = DrumKit("House Kit")
    
    # Add kick
    kit.add_sound(create_kick("deep_house"))
    
    # Add snare and clap
    kit.add_sound(create_snare("standard", reverb_amount=0.3))
    
    # Add hats
    kit.add_sound(create_hat("closed"))
    kit.add_sound(create_hat("open"))
    
    # Add percussion
    kit.add_sound(create_percussion("click", duration=0.1))
    
    # Add sub-bass
    kit.add_sound(create_sub_bass("basic", waveform="sine", base_freq=40))
    
    return kit

def create_techno_kit():
    """Create a techno drum kit"""
    kit = DrumKit("Techno Kit")
    
    # Add kick
    kit.add_sound(create_kick("techno"))
    
    # Add snare and clap
    kit.add_sound(create_snare("standard", body_freq=170, reverb_amount=0.15))
    
    # Add hats
    kit.add_sound(create_hat("closed", freq_range=(5000, 18000), decay=60))
    kit.add_sound(create_hat("open", freq_range=(4000, 16000), decay=15))
    
    # Add percussion
    kit.add_sound(create_percussion("blip", duration=0.1))
    kit.add_sound(create_percussion("microrhythm", duration=0.2))
    
    # Add sub-bass
    kit.add_sound(create_sub_bass("basic", waveform="triangle", base_freq=45, filter_freq=150))
    
    return kit

def create_dubstep_kit():
    """Create a dubstep drum kit"""
    kit = DrumKit("Dubstep Kit")
    
    # Add kick
    kit.add_sound(create_kick("vump", distortion=0.6))
    
    # Add snare and clap
    kit.add_sound(create_snare("dubstep"))
    kit.add_sound(create_snare("layered"))
    
    # Add hats
    kit.add_sound(create_hat("closed", decay=80))
    kit.add_sound(create_hat("open", decay=20))
    
    # Add percussion
    kit.add_sound(create_percussion("glitch", duration=0.15))
    kit.add_sound(create_percussion("artifact", duration=0.2))
    
    # Add sub-bass
    kit.add_sound(create_sub_bass("wobble", wobble_rate=4.0, format_type="dubstep", duration=1.0))
    
    return kit

def create_hardstyle_kit():
    """Create a hardstyle drum kit"""
    kit = DrumKit("Hardstyle Kit")
    
    # Add kick
    kit.add_sound(create_kick("hardstyle"))
    
    # Add snare and clap
    kit.add_sound(create_snare("hardstyle"))
    
    # Add hats
    kit.add_sound(create_hat("closed", decay=100, freq_range=(6000, 20000)))
    kit.add_sound(create_hat("open", decay=25, freq_range=(5000, 18000)))
    
    # Add percussion
    kit.add_sound(create_percussion("glitch", duration=0.1))
    
    # Add sub-bass
    kit.add_sound(create_sub_bass("808", format_type="modern", duration=1.0, base_freq=50))
    
    return kit

# Function to get a preset kit by name
def get_kit_by_name(name):
    """Get a preset kit by name"""
    kits = {
        "house": create_house_kit,
        "techno": create_techno_kit,
        "dubstep": create_dubstep_kit,
        "hardstyle": create_hardstyle_kit
    }
    
    kit_function = kits.get(name.lower())
    if kit_function:
        return kit_function()
    else:
        # Return a default kit if name not found
        return create_house_kit()
