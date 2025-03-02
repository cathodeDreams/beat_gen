import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

def generate_kick(sample_rate=44100, duration=0.5, 
                  freq_start=150, freq_end=60,
                  click_strength=0.5, volume=0.8):
    """
    Generate a kick drum sound
    
    Parameters:
    - sample_rate: Audio sample rate in Hz
    - duration: Length of the kick in seconds
    - freq_start: Starting frequency of the sine sweep in Hz
    - freq_end: Ending frequency of the sine sweep in Hz
    - click_strength: Amount of high-frequency click to add (0-1)
    - volume: Overall volume (0-1)
    
    Returns:
    - Numpy array of audio samples
    """
    # Create time array
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # Create amplitude envelope (quick attack, longer decay)
    amplitude = np.exp(-t * 12)  # Exponential decay
    
    # Create frequency envelope (rapid pitch drop)
    freq_t = np.exp(-t * 15)  # Exponential frequency change
    freq_envelope = freq_start + (freq_end - freq_start) * (1 - freq_t)
    
    # Generate sine wave with frequency envelope
    phase = np.cumsum(2 * np.pi * freq_envelope / sample_rate)
    sine_wave = np.sin(phase)
    
    # Apply amplitude envelope to sine wave
    kick_body = sine_wave * amplitude
    
    # Generate click (noise with high-pass filter for the attack)
    noise = np.random.uniform(-1, 1, num_samples)
    
    # Create a short click envelope
    click_env = np.exp(-t * 50)
    
    # High-pass filter the noise
    sos = signal.butter(2, 1000, 'hp', fs=sample_rate, output='sos')
    filtered_noise = signal.sosfilt(sos, noise)
    
    # Apply click envelope to filtered noise
    click = filtered_noise * click_env * click_strength
    
    # Combine kick body and click
    kick = kick_body + click
    
    # Normalize and apply volume
    kick = kick / np.max(np.abs(kick)) * volume
    
    return kick

def save_kick_sample(filename="kick.wav", sample_rate=44100):
    """Generate and save a kick drum sample as a WAV file"""
    kick = generate_kick(sample_rate=sample_rate)
    wavfile.write(filename, sample_rate, (kick * 32767).astype(np.int16))
    print(f"Kick drum sample saved to {filename}")
    
def plot_kick_waveform():
    """Generate and plot a kick drum waveform"""
    kick = generate_kick()
    plt.figure(figsize=(10, 6))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.plot(kick)
    plt.title('Kick Drum Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(kick, Fs=44100, NFFT=1024)
    plt.title('Kick Drum Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Generate and save a kick drum sample
    save_kick_sample()
    
    # Generate different variations
    punchy_kick = generate_kick(freq_start=200, freq_end=40, click_strength=0.7)
    deep_kick = generate_kick(freq_start=120, freq_end=30, click_strength=0.3)
    
    # Plot the waveform and spectrogram
    plot_kick_waveform()
