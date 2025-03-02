import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import random

def generate_glitch_percussion(
    sample_rate=44100,
    style="click",  # click, blip, grain, glitch, microrhythm, artifact
    duration=0.15,  # Short durations for electronic percussion
    volume=0.9
):
    """
    Generate digital/glitch percussion sounds
    
    Parameters:
    - sample_rate: Audio sample rate in Hz
    - style: Type of digital percussion to generate
    - duration: Length of the sound in seconds
    - volume: Overall volume
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # Start with silence
    sound = np.zeros(num_samples)
    
    if style == "click":
        # Super short sine clicks with varied frequencies
        clicks = 4 + random.randint(0, 3)
        for i in range(clicks):
            pos = random.uniform(0, duration * 0.7)
            freq = random.uniform(1000, 8000)
            length = random.uniform(0.002, 0.01)
            idx_range = (t >= pos) & (t < pos + length)
            sound[idx_range] += np.sin(2 * np.pi * freq * (t[idx_range] - pos)) * 0.8
    
    elif style == "blip":
        # FM synthesis "blips" with rapid pitch changes
        carrier_freq = random.uniform(200, 2000)
        mod_freq = random.uniform(50, 500)
        mod_index = random.uniform(2, 10)
        
        # Create FM synthesis
        modulator = np.sin(2 * np.pi * mod_freq * t) * mod_index
        sound = np.sin(2 * np.pi * carrier_freq * t + modulator)
        
        # Apply very short envelope
        env = np.exp(-t * random.uniform(30, 80))
        sound *= env
    
    elif style == "grain":
        # Granular synthesis - multiple tiny grains
        num_grains = random.randint(10, 30)
        for i in range(num_grains):
            pos = random.uniform(0, duration * 0.9)
            length = random.uniform(0.001, 0.02)
            freq = random.uniform(300, 7000)
            grain_env_rate = random.uniform(300, 1000)
            
            idx_range = (t >= pos) & (t < pos + length)
            if np.any(idx_range):
                grain_t = t[idx_range] - pos
                grain_env = np.exp(-grain_t * grain_env_rate)
                grain = np.sin(2 * np.pi * freq * grain_t) * grain_env
                sound[idx_range] += grain * random.uniform(0.2, 0.7)
    
    elif style == "glitch":
        # Digital glitch sounds with bit-crushing and sample rate effects
        base_sound = np.random.uniform(-1, 1, num_samples)
        
        # Apply bandpass filter
        sos = signal.butter(2, [random.uniform(500, 2000), random.uniform(3000, 8000)], 
                           'bp', fs=sample_rate, output='sos')
        filtered = signal.sosfilt(sos, base_sound)
        
        # Simulate bit-crushing
        bit_depth = random.randint(2, 5)
        step = 2.0 ** (bit_depth)
        sound = np.round(filtered * step) / step
        
        # Simulate sample rate reduction
        sr_reduce = random.randint(5, 15)
        for i in range(0, num_samples, sr_reduce):
            end = min(i + sr_reduce, num_samples)
            sound[i:end] = sound[i]
        
        # Apply envelope
        env = np.exp(-t * random.uniform(10, 30))
        sound *= env
    
    elif style == "microrhythm":
        # IDM-style micro-rhythmic patterns
        pattern_length = random.randint(4, 16)
        pattern = np.random.choice([0, 1], size=pattern_length, p=[0.6, 0.4])
        divisions = num_samples // pattern_length
        
        for i in range(pattern_length):
            if pattern[i] == 1:
                start = i * divisions
                end = (i + 1) * divisions
                
                # Random frequency for each hit
                freq = random.uniform(200, 5000)
                
                # Random envelope shape for each hit
                attack = random.uniform(0.1, 0.9)
                segment_t = np.linspace(0, 1, end - start)
                env = np.exp(-segment_t * random.uniform(10, 50))
                
                # Random waveform for each hit
                wave_type = random.choice(["sine", "square", "saw", "noise"])
                if wave_type == "sine":
                    wave = np.sin(2 * np.pi * freq * segment_t)
                elif wave_type == "square":
                    wave = np.sign(np.sin(2 * np.pi * freq * segment_t))
                elif wave_type == "saw":
                    wave = 2 * (freq * segment_t - np.floor(0.5 + freq * segment_t))
                else:  # noise
                    wave = np.random.uniform(-1, 1, end - start)
                    sos = signal.butter(2, [freq - 100, freq + 100], 
                                       'bp', fs=sample_rate, output='sos')
                    wave = signal.sosfilt(sos, wave)
                
                sound[start:end] += wave * env * random.uniform(0.3, 1.0)
    
    elif style == "artifact":
        # Digital artifacts and glitches
        # Start with noise bursts
        segments = random.randint(3, 8)
        segment_size = num_samples // segments
        
        for i in range(segments):
            if random.random() > 0.3:  # 70% chance of adding a segment
                start = i * segment_size
                end = (i + 1) * segment_size
                
                # Different types of artifacts
                artifact_type = random.choice(["noise", "tone", "crackle", "chirp"])
                
                if artifact_type == "noise":
                    # Filtered noise burst
                    noise = np.random.uniform(-1, 1, end - start)
                    cutoff = random.uniform(500, 8000)
                    sos = signal.butter(2, cutoff, 'lp', fs=sample_rate, output='sos')
                    sound[start:end] = signal.sosfilt(sos, noise) * 0.8
                
                elif artifact_type == "tone":
                    # Pure digital tone with abrupt start/stop
                    freq = random.uniform(300, 3000)
                    segment_t = np.linspace(0, (end - start) / sample_rate, end - start)
                    sound[start:end] = np.sin(2 * np.pi * freq * segment_t) * 0.7
                
                elif artifact_type == "crackle":
                    # Random impulses for digital crackle effect
                    crackle = np.zeros(end - start)
                    num_crackles = random.randint(3, 20)
                    positions = np.random.randint(0, end - start, num_crackles)
                    crackle[positions] = np.random.uniform(0.3, 1.0, num_crackles)
                    sound[start:end] = crackle
                
                elif artifact_type == "chirp":
                    # Frequency sweep (chirp)
                    segment_t = np.linspace(0, (end - start) / sample_rate, end - start)
                    f0 = random.uniform(100, 2000)
                    f1 = random.uniform(2000, 8000)
                    sound[start:end] = signal.chirp(segment_t, f0, segment_t[-1], f1) * 0.7
    
    # Apply random resonant filter for extra digital character
    if random.random() > 0.5:
        filter_freq = random.uniform(500, 5000)
        q = random.uniform(5, 20)  # High Q for resonance
        b, a = signal.iirpeak(filter_freq, q, sample_rate)
        sound = signal.lfilter(b, a, sound)
    
    # Normalize and apply volume
    if np.max(np.abs(sound)) > 0:  # Avoid division by zero
        sound = sound / np.max(np.abs(sound)) * volume
    
    return sound

def generate_ticky_sequence(
    sample_rate=44100,
    tempo=120,  # BPM
    num_steps=16,
    complexity=0.7,  # 0-1, higher means more complex rhythms
    duration=0.5  # seconds
):
    """Generate a sequence of glitchy percussion sounds"""
    # Calculate total length in samples
    beat_length = int(60 / tempo * sample_rate)
    sequence_length = beat_length * (num_steps / 4)  # 4 steps per beat
    
    # Create empty sequence
    sequence = np.zeros(int(sequence_length))
    
    # Probability matrix for different sounds based on complexity
    hit_probability = 0.2 + complexity * 0.5
    
    # Different sound types
    styles = ["click", "blip", "grain", "glitch", "microrhythm", "artifact"]
    
    # Generate rhythm pattern
    for step in range(num_steps):
        if random.random() < hit_probability:
            # Calculate position
            position = int(step * sequence_length / num_steps)
            
            # Choose random percussion style
            style = random.choice(styles)
            
            # Generate sound
            sound_duration = random.uniform(0.05, duration)
            sound = generate_glitch_percussion(
                sample_rate=sample_rate,
                style=style,
                duration=sound_duration
            )
            
            # Add to sequence
            end_pos = min(position + len(sound), len(sequence))
            add_length = end_pos - position
            sequence[position:end_pos] += sound[:add_length]
    
    # Normalize the sequence
    sequence = sequence / max(np.max(np.abs(sequence)), 0.001) * 0.9
    
    return sequence

def create_beep_boop_kit():
    """Create a collection of digital percussion sounds"""
    sample_rate = 44100
    kit = {}
    
    # Generate each type of sound
    for style in ["click", "blip", "grain", "glitch", "microrhythm", "artifact"]:
        sounds = []
        # Create multiple variations of each type
        for i in range(5):
            sound = generate_glitch_percussion(style=style, duration=random.uniform(0.1, 0.3))
            sounds.append(sound)
        kit[style] = sounds
    
    return kit, sample_rate

def save_beep_boop_kit(prefix="beep_boop"):
    """Save a kit of digital percussion sounds"""
    kit, sample_rate = create_beep_boop_kit()
    
    for style, sounds in kit.items():
        for i, sound in enumerate(sounds):
            filename = f"{prefix}_{style}_{i+1}.wav"
            wavfile.write(filename, sample_rate, (sound * 32767).astype(np.int16))
            print(f"Saved {filename}")

def save_ticky_sequence(filename="ticky_sequence.wav", tempo=120, complexity=0.7):
    """Generate and save a sequence of ticky sounds"""
    sample_rate = 44100
    sequence = generate_ticky_sequence(
        sample_rate=sample_rate,
        tempo=tempo,
        complexity=complexity
    )
    
    wavfile.write(filename, sample_rate, (sequence * 32767).astype(np.int16))
    print(f"Saved sequence to {filename}")
    
    return sequence

def visualize_glitch_sound(style="click"):
    """Visualize a glitch percussion sound"""
    sample_rate = 44100
    sound = generate_glitch_percussion(style=style)
    
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.plot(sound)
    plt.title(f'{style.title()} Digital Percussion Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(alpha=0.3)
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(sound, NFFT=256, Fs=sample_rate, noverlap=128, cmap='plasma')
    plt.title(f'{style.title()} Digital Percussion Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.ylim(0, 20000)  # Show full frequency range
    
    plt.tight_layout()
    plt.show()

def create_custom_beep_boop(
    clicks=True,
    resonance=True,
    glitchy=True,
    duration=0.2
):
    """Create a custom beep boop sound with specific characteristics"""
    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    sound = np.zeros(num_samples)
    
    # Add clicks if requested
    if clicks:
        num_clicks = random.randint(3, 8)
        for i in range(num_clicks):
            pos = random.uniform(0, duration * 0.8)
            length = random.uniform(0.001, 0.01)
            freq = random.uniform(2000, 7000)
            idx_range = (t >= pos) & (t < pos + length)
            if np.any(idx_range):
                click_t = t[idx_range] - pos
                click = np.sin(2 * np.pi * freq * click_t) * np.exp(-click_t * 300)
                sound[idx_range] += click * 0.5
    
    # Add resonant tones if requested
    if resonance:
        # Primary tone
        freq = random.uniform(400, 2000)
        env = np.exp(-t * random.uniform(10, 40))
        tone = np.sin(2 * np.pi * freq * t) * env * 0.6
        
        # Add some harmonics or modulation
        if random.random() > 0.5:
            # Harmonics
            harmonic = np.sin(2 * np.pi * freq * 1.5 * t) * env * 0.3
            tone += harmonic
        else:
            # FM modulation
            mod_freq = random.uniform(30, 100)
            mod_index = random.uniform(1, 3)
            modulator = np.sin(2 * np.pi * mod_freq * t) * mod_index
            tone = np.sin(2 * np.pi * freq * t + modulator) * env * 0.6
        
        sound += tone
    
    # Add glitchy artifacts if requested
    if glitchy:
        if random.random() > 0.5:
            # Bit reduction
            bit_depth = random.randint(2, 4)
            step = 2.0 ** (bit_depth)
            sound = np.round(sound * step) / step
        
        if random.random() > 0.5:
            # Sample rate reduction effect
            sr_reduce = random.randint(4, 10)
            for i in range(0, num_samples, sr_reduce):
                end = min(i + sr_reduce, num_samples)
                sound[i:end] = sound[i]
        
        if random.random() > 0.7:
            # Random amplitude jumps
            num_jumps = random.randint(2, 5)
            for i in range(num_jumps):
                pos = random.randint(0, num_samples - 1)
                width = random.randint(1, 100)
                end = min(pos + width, num_samples)
                sound[pos:end] *= random.uniform(-1.5, 1.5)
    
    # Apply a filter for extra character
    filter_type = random.choice(['lp', 'hp', 'bp', 'peak'])
    
    if filter_type == 'lp':
        sos = signal.butter(2, random.uniform(1000, 8000), 'lp', fs=sample_rate, output='sos')
        sound = signal.sosfilt(sos, sound)
    elif filter_type == 'hp':
        sos = signal.butter(2, random.uniform(500, 2000), 'hp', fs=sample_rate, output='sos')
        sound = signal.sosfilt(sos, sound)
    elif filter_type == 'bp':
        sos = signal.butter(2, [random.uniform(500, 2000), random.uniform(3000, 8000)], 
                           'bp', fs=sample_rate, output='sos')
        sound = signal.sosfilt(sos, sound)
    else:  # peak
        filter_freq = random.uniform(500, 5000)
        q = random.uniform(5, 20)
        b, a = signal.iirpeak(filter_freq, q, sample_rate)
        sound = signal.lfilter(b, a, sound)
    
    # Normalize
    if np.max(np.abs(sound)) > 0:
        sound = sound / np.max(np.abs(sound)) * 0.9
    
    return sound, sample_rate

# Example usage:
if __name__ == "__main__":
    # Save a kit of various glitch sounds
    save_beep_boop_kit()
    
    # Generate and save some sequences
    save_ticky_sequence("ticky_slow.wav", tempo=90, complexity=0.5)
    save_ticky_sequence("ticky_fast.wav", tempo=160, complexity=0.8)
    
    # Visualize different glitch styles
    for style in ["click", "blip", "grain", "glitch", "microrhythm", "artifact"]:
        visualize_glitch_sound(style)
