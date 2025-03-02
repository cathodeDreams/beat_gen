import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

def generate_edm_snare(
    sample_rate=44100,
    duration=0.5,
    body_freq=180,          # Frequency of the tonal component
    noise_amount=0.8,       # Amount of noise component
    snap_amount=0.6,        # Amount of snap/transient
    body_amount=0.4,        # Amount of tonal body
    body_decay=15,          # How quickly the body decays
    noise_decay=8,          # How quickly the noise decays
    distortion=0.2,         # Distortion amount
    highpass_freq=200,      # Highpass filter cutoff
    bandpass_low=400,       # Bandpass lower cutoff
    bandpass_high=6000,     # Bandpass upper cutoff
    reverb_amount=0.2,      # Reverb amount
    volume=0.9
):
    """
    Generate an EDM snare drum
    
    Parameters:
    - sample_rate: Audio sample rate in Hz
    - duration: Length of the snare in seconds
    - body_freq: Frequency of the tonal body component
    - noise_amount: Amount of noise component
    - snap_amount: Amount of initial snap/attack
    - body_amount: Amount of tonal body
    - body_decay: Decay rate of the tonal body
    - noise_decay: Decay rate of the noise component
    - distortion: Amount of distortion/saturation
    - highpass_freq: Highpass filter cutoff for overall sound
    - bandpass_low: Lower cutoff for bandpass on noise component
    - bandpass_high: Upper cutoff for bandpass on noise component
    - reverb_amount: Amount of reverb to add
    - volume: Overall volume
    
    Returns:
    - Numpy array of audio samples
    """
    # Create time array
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # Generate tonal body component (typically a sine or triangle wave)
    # Use triangle wave for more harmonics
    body_wave = 2 * np.abs(2 * (t * body_freq - np.floor(t * body_freq + 0.5))) - 1
    
    # Apply amplitude envelope to body
    body_env = np.exp(-t * body_decay)
    snare_body = body_wave * body_env * body_amount
    
    # Generate noise component (the "shhh" part)
    noise = np.random.uniform(-1, 1, num_samples)
    
    # Apply bandpass filter to noise
    sos_bandpass = signal.butter(2, [bandpass_low, bandpass_high], 'bp', fs=sample_rate, output='sos')
    filtered_noise = signal.sosfilt(sos_bandpass, noise)
    
    # Apply envelope to noise
    noise_env = np.exp(-t * noise_decay)
    snare_noise = filtered_noise * noise_env * noise_amount
    
    # Generate snap component (the initial "crack")
    snap_noise = np.random.uniform(-1, 1, num_samples)
    
    # Highpass filter the snap noise (higher than the main noise)
    sos_snap = signal.butter(4, 1200, 'hp', fs=sample_rate, output='sos')
    filtered_snap = signal.sosfilt(sos_snap, snap_noise)
    
    # Apply very fast decay to snap
    snap_env = np.exp(-t * 60)
    snare_snap = filtered_snap * snap_env * snap_amount
    
    # Combine all components
    snare = snare_body + snare_noise + snare_snap
    
    # Apply distortion for that EDM crunch
    if distortion > 0:
        # Soft clipping distortion
        snare = np.tanh(snare * (1 + distortion * 3)) / (1 + distortion)
    
    # Apply highpass filter to remove unwanted low frequencies
    sos_highpass = signal.butter(4, highpass_freq, 'hp', fs=sample_rate, output='sos')
    snare = signal.sosfilt(sos_highpass, snare)
    
    # Add simple reverb if requested
    if reverb_amount > 0:
        # Create a simple impulse response
        reverb_length = int(sample_rate * 0.1)  # 100ms reverb
        impulse_response = np.exp(-np.linspace(0, 8, reverb_length))
        impulse_response = impulse_response / np.sum(impulse_response)  # Normalize
        
        # Convolve with signal (apply reverb)
        reverb_signal = np.convolve(snare, impulse_response)[:num_samples]
        snare = snare + reverb_signal * reverb_amount
    
    # Normalize and apply volume
    snare = snare / np.max(np.abs(snare)) * volume
    
    return snare

def generate_specific_edm_snares():
    """Generate different types of EDM snares"""
    # Standard EDM snare
    standard = generate_edm_snare(
        body_freq=180,
        noise_amount=0.8,
        snap_amount=0.6,
        body_amount=0.4,
        distortion=0.2,
        reverb_amount=0.2
    )
    
    # Trap snare (sharper, shorter, more snap)
    trap = generate_edm_snare(
        body_freq=220,
        noise_amount=0.7,
        snap_amount=0.9,
        body_amount=0.3,
        body_decay=20,
        noise_decay=12,
        distortion=0.3,
        bandpass_high=8000,
        reverb_amount=0.15
    )
    
    # Dubstep snare (more aggressive, distorted)
    dubstep = generate_edm_snare(
        body_freq=150,
        noise_amount=0.9,
        snap_amount=0.8,
        body_amount=0.6,
        body_decay=10,
        noise_decay=7,
        distortion=0.5,
        bandpass_low=300,
        bandpass_high=7000,
        reverb_amount=0.3
    )
    
    # Future bass snare (bright, airy, reverby)
    future_bass = generate_edm_snare(
        body_freq=200,
        noise_amount=0.85,
        snap_amount=0.7,
        body_amount=0.35,
        body_decay=12,
        noise_decay=6,
        distortion=0.1,
        bandpass_low=500,
        bandpass_high=10000,
        reverb_amount=0.4
    )
    
    # Hardstyle/Hardcore snare (very aggressive)
    hardstyle = generate_edm_snare(
        body_freq=180,
        noise_amount=0.95,
        snap_amount=0.9,
        body_amount=0.5,
        body_decay=8,
        noise_decay=5,
        distortion=0.7,
        bandpass_low=300,
        bandpass_high=9000,
        reverb_amount=0.25
    )
    
    return {
        "standard": standard,
        "trap": trap,
        "dubstep": dubstep,
        "future_bass": future_bass,
        "hardstyle": hardstyle
    }

def save_edm_snare(snare_type="dubstep", filename=None):
    """Generate and save an EDM snare as a WAV file"""
    snares = generate_specific_edm_snares()
    
    if snare_type not in snares:
        snare_type = "dubstep"  # Default to dubstep style
    
    snare = snares[snare_type]
    
    if filename is None:
        filename = f"{snare_type}_snare.wav"
    
    wavfile.write(filename, 44100, (snare * 32767).astype(np.int16))
    print(f"{snare_type.title()} snare drum saved to {filename}")
    
    return snare

def layered_snare():
    """Create a layered snare with multiple components for extra depth"""
    # Main snare - dubstep style
    main = generate_edm_snare(
        body_freq=160,
        noise_amount=0.85,
        snap_amount=0.7,
        body_amount=0.5,
        distortion=0.3,
        reverb_amount=0.25
    )
    
    # Bright layer - for extra snap
    bright = generate_edm_snare(
        body_freq=220,
        noise_amount=0.6,
        snap_amount=0.9,
        body_amount=0.2,
        body_decay=25,
        noise_decay=15,
        distortion=0.1,
        highpass_freq=1500,
        volume=0.5
    )
    
    # Reverb tail - for ambience
    tail = generate_edm_snare(
        body_freq=180,
        noise_amount=0.7,
        snap_amount=0.3,
        body_amount=0.3,
        noise_decay=4,
        reverb_amount=0.8,
        highpass_freq=400,
        volume=0.4
    )
    
    # Combine layers (with normalization)
    layered = main + bright + tail
    layered = layered / np.max(np.abs(layered)) * 0.9
    
    return layered

def visualize_snare(snare_type="dubstep"):
    """Visualize the waveform and spectrogram of an EDM snare"""
    snares = generate_specific_edm_snares()
    
    if snare_type == "layered":
        snare = layered_snare()
    elif snare_type not in snares:
        snare_type = "dubstep"
        snare = snares[snare_type]
    else:
        snare = snares[snare_type]
    
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.plot(snare)
    plt.title(f'{snare_type.title()} EDM Snare Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(alpha=0.3)
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(snare, NFFT=512, Fs=44100, noverlap=256, cmap='viridis')
    plt.title(f'{snare_type.title()} EDM Snare Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.ylim(0, 15000)  # Show more high frequencies for snare
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Save all snare types
    for snare_type in ["standard", "trap", "dubstep", "future_bass", "hardstyle"]:
        save_edm_snare(snare_type)
    
    # Save a special layered snare
    layered = layered_snare()
    wavfile.write("layered_snare.wav", 44100, (layered * 32767).astype(np.int16))
    
    # Visualize the layered snare
    visualize_snare("layered")
