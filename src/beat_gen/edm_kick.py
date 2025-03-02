import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

def generate_edm_kick(
    sample_rate=44100,
    duration=0.6,
    freq_start=180,
    freq_end=40,
    pitch_bend_speed=18,    # Higher = faster pitch drop
    punch_amount=1.2,       # Transient punch enhancement
    distortion=0.3,         # Distortion amount
    texture_amount=0.25,    # Amount of texture noise
    body_resonance=0.4,     # Resonance in the body
    volume=0.9
):
    """
    Generate a textured EDM kick drum with VUMP characteristics
    
    Parameters:
    - sample_rate: Audio sample rate in Hz
    - duration: Length of the kick in seconds
    - freq_start: Starting frequency of the sine sweep in Hz
    - freq_end: Ending frequency of the sine sweep in Hz
    - pitch_bend_speed: How quickly the pitch drops (higher = faster)
    - punch_amount: Amount of transient punch
    - distortion: Amount of distortion/saturation to add
    - texture_amount: Amount of textural elements to add
    - body_resonance: Resonance amount in the body of the kick
    - volume: Overall volume
    
    Returns:
    - Numpy array of audio samples
    """
    # Create time array
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # Create complex amplitude envelopes for that EDM pump
    # Main body envelope (slower decay)
    body_env = np.exp(-t * 8)
    
    # Punch envelope (very fast attack, quick decay)
    punch_env = np.exp(-t * 50) * punch_amount
    
    # Create a more complex frequency envelope for the "VUMP" effect
    # Use a combination of exponential and logarithmic curves
    freq_t = 1 - np.exp(-t * pitch_bend_speed)
    freq_envelope = freq_start * np.exp(-freq_t * 2) + freq_end
    
    # Generate sine wave with frequency envelope for the main body
    phase = np.cumsum(2 * np.pi * freq_envelope / sample_rate)
    sine_wave = np.sin(phase)
    
    # Add some resonance to the body (slight phase offset)
    if body_resonance > 0:
        resonance = np.sin(phase + 0.2) * np.exp(-t * 12) * body_resonance
        sine_wave += resonance
    
    # Apply main envelope to sine wave
    kick_body = sine_wave * body_env
    
    # Generate punch component (higher frequency with very quick decay)
    punch_freq = np.linspace(freq_start * 1.5, freq_start, num_samples)
    punch_phase = np.cumsum(2 * np.pi * punch_freq / sample_rate)
    punch = np.sin(punch_phase) * punch_env
    
    # Generate click/transient for attack
    noise = np.random.uniform(-1, 1, num_samples)
    click_env = np.exp(-t * 80)  # Very quick decay
    
    # High-pass filter the noise (for click)
    sos = signal.butter(2, 1500, 'hp', fs=sample_rate, output='sos')
    click = signal.sosfilt(sos, noise) * click_env * 0.7
    
    # Generate texture layer (filtered noise with medium decay)
    texture_env = np.exp(-t * 25) * texture_amount
    texture_noise = np.random.uniform(-1, 1, num_samples)
    
    # Band-pass filter the texture noise
    sos_texture = signal.butter(2, [200, 4000], 'bp', fs=sample_rate, output='sos')
    texture = signal.sosfilt(sos_texture, texture_noise) * texture_env
    
    # Combine all components
    kick = kick_body + punch + click + texture
    
    # Apply distortion/saturation for that EDM texture
    if distortion > 0:
        # Soft clipping distortion
        kick = np.tanh(kick * (1 + distortion * 5)) / (1 + distortion)
    
    # Apply a subtle EQ (boost low end for more VUMP)
    sos_eq = signal.butter(2, 80, 'lp', fs=sample_rate, output='sos')
    eq_boost = signal.sosfilt(sos_eq, kick) * 0.3
    kick += eq_boost
    
    # Normalize and apply volume
    kick = kick / np.max(np.abs(kick)) * volume
    
    return kick

def generate_specific_edm_kicks():
    """Generate different types of EDM kicks"""
    # Standard EDM kick
    standard = generate_edm_kick(
        freq_start=180, 
        freq_end=40, 
        pitch_bend_speed=18,
        punch_amount=1.2, 
        distortion=0.3
    )
    
    # Deep House kick (lower, rounder)
    deep_house = generate_edm_kick(
        freq_start=140, 
        freq_end=35, 
        pitch_bend_speed=12, 
        punch_amount=0.8, 
        distortion=0.2, 
        texture_amount=0.1,
        body_resonance=0.5
    )
    
    # Hardstyle kick (more distorted, aggressive)
    hardstyle = generate_edm_kick(
        freq_start=200, 
        freq_end=45, 
        pitch_bend_speed=25, 
        punch_amount=1.5, 
        distortion=0.7, 
        texture_amount=0.4,
        body_resonance=0.7
    )
    
    # Dubstep/VUMP kick (very punchy with texture)
    vump = generate_edm_kick(
        freq_start=160, 
        freq_end=30, 
        pitch_bend_speed=20, 
        punch_amount=1.8, 
        distortion=0.5, 
        texture_amount=0.3,
        body_resonance=0.6
    )
    
    return {
        "standard": standard,
        "deep_house": deep_house,
        "hardstyle": hardstyle,
        "vump": vump
    }

def save_edm_kick(kick_type="vump", filename=None):
    """Generate and save an EDM kick drum as a WAV file"""
    kicks = generate_specific_edm_kicks()
    
    if kick_type not in kicks:
        kick_type = "vump"  # Default to VUMP style
    
    kick = kicks[kick_type]
    
    if filename is None:
        filename = f"{kick_type}_kick.wav"
    
    wavfile.write(filename, 44100, (kick * 32767).astype(np.int16))
    print(f"{kick_type.title()} kick drum saved to {filename}")
    
    return kick

def visualize_kick(kick_type="vump"):
    """Visualize the waveform and spectrogram of an EDM kick"""
    kicks = generate_specific_edm_kicks()
    
    if kick_type not in kicks:
        kick_type = "vump"
    
    kick = kicks[kick_type]
    
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.plot(kick)
    plt.title(f'{kick_type.title()} EDM Kick Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(alpha=0.3)
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(kick, NFFT=512, Fs=44100, noverlap=256, cmap='inferno')
    plt.title(f'{kick_type.title()} EDM Kick Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.ylim(0, 5000)  # Limit frequency range to focus on important parts
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Save all kick types
    for kick_type in ["vump", "standard", "deep_house", "hardstyle"]:
        save_edm_kick(kick_type)
    
    # Visualize the VUMP kick
    visualize_kick("vump")
