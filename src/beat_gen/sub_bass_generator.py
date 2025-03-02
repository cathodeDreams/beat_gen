import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

def generate_sub_bass(
    sample_rate=44100,
    duration=2.0,
    base_freq=40,          # Sub-bass frequency in Hz
    waveform="sine",       # sine, triangle, square, saw
    saturation=0.0,        # 0.0-1.0 amount of saturation/distortion
    filter_freq=120,       # Lowpass filter cutoff
    filter_res=0.1,        # Filter resonance (0-1)
    volume=0.9
):
    """
    Generate a basic sub-bass tone
    
    Parameters:
    - sample_rate: Audio sample rate in Hz
    - duration: Length of the sound in seconds
    - base_freq: Fundamental frequency of the sub-bass (typically 30-60 Hz)
    - waveform: Type of waveform to use
    - saturation: Amount of saturation to add
    - filter_freq: Lowpass filter cutoff frequency
    - filter_res: Filter resonance amount
    - volume: Overall volume
    
    Returns:
    - Numpy array of audio samples
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # Generate base waveform
    if waveform == "sine":
        wave = np.sin(2 * np.pi * base_freq * t)
    elif waveform == "triangle":
        wave = 2 * np.abs(2 * (t * base_freq - np.floor(t * base_freq + 0.5))) - 1
    elif waveform == "square":
        wave = np.sign(np.sin(2 * np.pi * base_freq * t))
    elif waveform == "saw":
        wave = 2 * (t * base_freq - np.floor(t * base_freq))
    else:  # Default to sine
        wave = np.sin(2 * np.pi * base_freq * t)
    
    # Apply saturation/distortion if requested
    if saturation > 0:
        # Soft clipping distortion
        wave = np.tanh(wave * (1 + saturation * 5)) / (1 + saturation)
    
    # Apply lowpass filter to shape the sound
    if filter_freq > 0:
        # Convert resonance to Q factor (0-1 resonance to approximate Q)
        q = 0.5 + filter_res * 10  # Q from ~0.5 to ~10.5
        
        # Create a lowpass filter with resonance
        b, a = signal.iirfilter(
            2, 
            filter_freq / (sample_rate/2),
            btype='lowpass',
            ftype='butter',
            output='ba'
        )
        
        # Apply filter
        wave = signal.lfilter(b, a, wave)
    
    # Normalize and apply volume
    wave = wave / np.max(np.abs(wave)) * volume
    
    return wave

def apply_amplitude_modulation(
    wave,
    sample_rate=44100,
    mod_freq=0.5,           # Modulation frequency in Hz
    mod_depth=0.5,          # Modulation depth (0-1)
    mod_shape="sine",       # sine, triangle, square, saw, custom
    custom_shape=None       # Custom envelope array
):
    """Apply amplitude modulation to sub-bass"""
    duration = len(wave) / sample_rate
    t = np.linspace(0, duration, len(wave))
    
    # Generate modulator wave
    if custom_shape is not None and len(custom_shape) == len(wave):
        # Use provided custom shape
        modulator = custom_shape
    else:
        # Generate standard waveshape
        if mod_shape == "sine":
            modulator = np.sin(2 * np.pi * mod_freq * t)
        elif mod_shape == "triangle":
            modulator = 2 * np.abs(2 * (t * mod_freq - np.floor(t * mod_freq + 0.5))) - 1
        elif mod_shape == "square":
            modulator = np.sign(np.sin(2 * np.pi * mod_freq * t))
        elif mod_shape == "saw":
            modulator = 2 * (t * mod_freq - np.floor(t * mod_freq)) - 1
        else:  # Default to sine
            modulator = np.sin(2 * np.pi * mod_freq * t)
    
    # Normalize modulator to 0-1 range
    modulator = (modulator + 1) / 2
    
    # Scale modulator based on depth
    modulator = 1 - (mod_depth * modulator)
    
    # Apply modulation
    modulated = wave * modulator
    
    # Normalize
    if np.max(np.abs(modulated)) > 0:
        modulated = modulated / np.max(np.abs(modulated)) * np.max(np.abs(wave))
    
    return modulated

def apply_filter_modulation(
    wave,
    sample_rate=44100,
    filter_type="lowpass",  # lowpass, bandpass
    base_freq=120,          # Base filter frequency
    mod_freq=0.5,           # Modulation frequency in Hz
    mod_depth=100,          # Frequency modulation range in Hz
    resonance=3.0           # Filter resonance (Q factor)
):
    """Apply filter modulation to sub-bass"""
    duration = len(wave) / sample_rate
    t = np.linspace(0, duration, len(wave))
    
    # Create filter modulation envelope
    mod_env = (np.sin(2 * np.pi * mod_freq * t) + 1) / 2  # 0-1 range
    
    # Calculate filter frequency over time
    filter_env = base_freq + mod_env * mod_depth
    
    # Apply time-varying filter
    output = np.zeros_like(wave)
    
    # Process in small chunks with different filter settings
    chunk_size = int(sample_rate / 100)  # ~10ms chunks
    for i in range(0, len(wave), chunk_size):
        end = min(i + chunk_size, len(wave))
        chunk = wave[i:end]
        
        # Get average filter frequency for this chunk
        avg_freq = np.mean(filter_env[i:end])
        
        # Create filter
        if filter_type == "lowpass":
            b, a = signal.iirfilter(
                2, 
                avg_freq / (sample_rate/2),
                btype='lowpass',
                ftype='butter',
                output='ba'
            )
        elif filter_type == "bandpass":
            b, a = signal.iirfilter(
                2, 
                [max(20, avg_freq - 20) / (sample_rate/2), 
                 min(20000, avg_freq + 20) / (sample_rate/2)],
                btype='bandpass',
                ftype='butter',
                output='ba'
            )
        else:  # Default to lowpass
            b, a = signal.iirfilter(
                2, 
                avg_freq / (sample_rate/2),
                btype='lowpass',
                ftype='butter',
                output='ba'
            )
        
        # Apply filter to chunk
        filtered_chunk = signal.lfilter(b, a, chunk)
        output[i:end] = filtered_chunk
    
    return output

def apply_frequency_modulation(
    sample_rate=44100,
    duration=2.0,
    base_freq=40,           # Base frequency
    mod_freq=0.25,          # Modulation frequency
    mod_depth=10,           # Frequency deviation in Hz
    waveform="sine"         # Carrier waveform
):
    """Create frequency modulated sub-bass"""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # Create modulator
    modulator = np.sin(2 * np.pi * mod_freq * t) * mod_depth
    
    # Create instantaneous frequency
    inst_freq = base_freq + modulator
    
    # Integrate frequency to get phase
    phase = np.cumsum(2 * np.pi * inst_freq / sample_rate)
    
    # Generate waveform with FM
    if waveform == "sine":
        wave = np.sin(phase)
    elif waveform == "triangle":
        wave = 2 * np.abs(2 * (phase/(2*np.pi) - np.floor(phase/(2*np.pi) + 0.5))) - 1
    elif waveform == "square":
        wave = np.sign(np.sin(phase))
    elif waveform == "saw":
        wave = 2 * (phase/(2*np.pi) - np.floor(phase/(2*np.pi))) - 1
    else:  # Default to sine
        wave = np.sin(phase)
    
    return wave

def create_wobble_bass(
    sample_rate=44100,
    duration=4.0,
    base_freq=40,
    wobble_rate=4.0,        # Wobble speed in Hz
    wobble_depth=0.9,       # 0-1 wobble intensity
    distortion=0.3,         # 0-1 distortion amount
    format_type="dubstep"   # dubstep, neuro, future
):
    """Create a wobble bass sound with filter modulation"""
    # Generate base sub
    sub = generate_sub_bass(
        sample_rate=sample_rate,
        duration=duration,
        base_freq=base_freq,
        waveform="saw",     # Start with saw for rich harmonics
        saturation=0.2,
        filter_freq=2000,   # High initial cutoff to be modulated
        volume=0.9
    )
    
    # Create time array for modulation (needed for all format types)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Different wobble characteristics based on format
    if format_type == "dubstep":
        # Classic dubstep wobble (square LFO, resonant filter)
        wobble = apply_filter_modulation(
            sub,
            sample_rate=sample_rate,
            filter_type="lowpass",
            base_freq=100,
            mod_freq=wobble_rate,
            mod_depth=2000,
            resonance=8.0
        )
        
        # Add distortion
        wobble = np.tanh(wobble * (1 + distortion * 3)) / (1 + distortion)
        
    elif format_type == "neuro":
        # Neurobass style (more complex modulation, aggressive)
        # Create a more complex modulation shape
        mod1 = np.sin(2 * np.pi * wobble_rate * t)
        mod2 = np.sin(2 * np.pi * wobble_rate * 1.5 * t)
        complex_mod = ((mod1 + mod2 * 0.5) / 1.5 + 1) / 2  # 0-1 range
        
        # Apply custom envelope filter modulation
        wobble = np.zeros_like(sub)
        chunk_size = int(sample_rate / 100)  # ~10ms chunks
        
        for i in range(0, len(sub), chunk_size):
            end = min(i + chunk_size, len(sub))
            chunk = sub[i:end]
            
            # Calculate filter frequency for this chunk
            mod_val = np.mean(complex_mod[i:end])
            filter_freq = 80 + mod_val * 4000
            
            # Create filter
            b, a = signal.iirfilter(
                2, 
                filter_freq / (sample_rate/2),
                btype='lowpass',
                ftype='butter',
                output='ba'
            )
            
            # Apply filter
            wobble[i:end] = signal.lfilter(b, a, chunk)
        
        # Add distortion (more for neuro)
        wobble = np.tanh(wobble * (1 + distortion * 5)) / (1 + distortion * 0.8)
        
    elif format_type == "future":
        # Future bass style (smoother, more sine-like modulation)
        wobble = apply_filter_modulation(
            sub,
            sample_rate=sample_rate,
            filter_type="lowpass",
            base_freq=150,
            mod_freq=wobble_rate / 2,  # Slower modulation
            mod_depth=1500,
            resonance=4.0
        )
        
        # Subtle saturation
        wobble = np.tanh(wobble * (1 + distortion * 2)) / (1 + distortion)
        
        # Add amplitude modulation for pumping effect
        amp_mod = np.sin(2 * np.pi * wobble_rate / 2 * t)
        amp_mod = (amp_mod + 1) / 2  # 0-1 range
        amp_mod = 1 - (0.3 * amp_mod)  # Scale modulation depth
        wobble = wobble * amp_mod
        
    else:  # Default to dubstep
        wobble = apply_filter_modulation(
            sub,
            sample_rate=sample_rate,
            filter_type="lowpass",
            base_freq=100,
            mod_freq=wobble_rate,
            mod_depth=2000,
            resonance=8.0
        )
    
    # Normalize
    wobble = wobble / np.max(np.abs(wobble)) * 0.95
    
    return wobble

def create_808_bass(
    sample_rate=44100,
    duration=2.0,
    base_freq=40,
    pitch_decay=3.0,         # Higher = faster pitch drop
    amp_decay=2.0,           # Higher = faster amplitude decay
    distortion=0.2,          # 0-1 distortion amount
    format_type="trap"       # trap, modern, clean
):
    """Create an 808-style sub bass with pitch envelope"""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # Create pitch envelope (downward slide characteristic of 808)
    pitch_env = np.exp(-t * pitch_decay)
    freq_envelope = base_freq * 2 + (base_freq - base_freq * 2) * (1 - pitch_env)
    
    # Create amplitude envelope
    amp_env = np.exp(-t * amp_decay)
    
    # Generate sine oscillator with pitch envelope
    phase = np.cumsum(2 * np.pi * freq_envelope / sample_rate)
    wave = np.sin(phase)
    
    # Apply amplitude envelope
    wave = wave * amp_env
    
    # Apply different processing based on format
    if format_type == "trap":
        # Typical trap 808: mild distortion, subtle harmonics
        wave = np.tanh(wave * (1 + distortion * 3)) / (1 + distortion)
        
        # Add subtle harmonics
        harmonics = np.sin(phase * 2) * 0.1 + np.sin(phase * 3) * 0.05
        wave += harmonics * amp_env
        
    elif format_type == "modern":
        # Modern 808: more distortion, stronger harmonics
        wave = np.tanh(wave * (1 + distortion * 5)) / (1 + distortion * 0.8)
        
        # Add stronger harmonics
        harmonics = np.sin(phase * 2) * 0.2 + np.sin(phase * 3) * 0.1 + np.sin(phase * 4) * 0.05
        wave += harmonics * amp_env
        
        # Add subtle sub-harmonic
        sub_harmonic = np.sin(phase * 0.5) * 0.15 * amp_env
        wave += sub_harmonic
        
    elif format_type == "clean":
        # Clean 808: minimal distortion, pure
        wave = np.tanh(wave * (1 + distortion * 1.5)) / (1 + distortion * 0.5)
        
        # Just a touch of 2nd harmonic
        harmonic = np.sin(phase * 2) * 0.05 * amp_env
        wave += harmonic
    
    # Normalize
    wave = wave / np.max(np.abs(wave)) * 0.95
    
    return wave

def create_reese_bass(
    sample_rate=44100,
    duration=2.0,
    base_freq=40,
    detune_amount=0.1,       # Detune factor for oscillators
    num_oscillators=3,       # Number of detuned oscillators
    movement=0.2,            # 0-1 amount of movement in detuning
    movement_rate=0.5,       # Rate of movement in Hz
    filter_freq=800,         # Filter cutoff
    filter_res=2.0,          # Filter resonance
    distortion=0.3           # Distortion amount
):
    """Create a Reese bass (layered detuned oscillators)"""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # Create movement modulator if needed
    if movement > 0:
        mod = np.sin(2 * np.pi * movement_rate * t) * movement
    else:
        mod = np.zeros_like(t)
    
    # Initialize combined wave
    combined = np.zeros(num_samples)
    
    # Generate detuned oscillators
    for i in range(num_oscillators):
        # Calculate detune factor with movement
        detune_factor = 1.0 + detune_amount * (i / (num_oscillators - 1) - 0.5) * 2
        detune_factor += mod * 0.01 * (i - num_oscillators // 2)
        
        # Generate frequency envelope with slight movement
        freq = base_freq * detune_factor
        
        # Generate saw oscillator
        osc = 2 * (t * freq - np.floor(t * freq)) - 1
        
        # Add to combined signal
        combined += osc / num_oscillators
    
    # Apply filter
    sos = signal.butter(2, filter_freq, 'lp', fs=sample_rate, output='sos')
    filtered = signal.sosfilt(sos, combined)
    
    # Add distortion
    bass = np.tanh(filtered * (1 + distortion * 3)) / (1 + distortion)
    
    # Apply amplitude envelope (subtle fade in/out)
    env = np.ones_like(bass)
    fade_samples = int(sample_rate * 0.01)  # 10ms fade
    env[:fade_samples] = np.linspace(0, 1, fade_samples)
    env[-fade_samples:] = np.linspace(1, 0, fade_samples)
    bass *= env
    
    # Normalize
    bass = bass / np.max(np.abs(bass)) * 0.95
    
    return bass

def visualize_sub_bass(wave, sample_rate=44100, title="Sub Bass Waveform"):
    """Visualize sub-bass waveform and spectrum"""
    plt.figure(figsize=(12, 10))
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    time = np.linspace(0, len(wave) / sample_rate, len(wave))
    plt.plot(time, wave)
    plt.title(f'{title} - Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(alpha=0.3)
    
    # Plot spectrogram
    plt.subplot(3, 1, 2)
    plt.specgram(wave, NFFT=2048, Fs=sample_rate, noverlap=1024, cmap='inferno')
    plt.title(f'{title} - Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.ylim(0, 1000)  # Focus on sub-bass range
    
    # Plot frequency spectrum
    plt.subplot(3, 1, 3)
    spectrum = np.abs(np.fft.rfft(wave))
    freq = np.fft.rfftfreq(len(wave), 1/sample_rate)
    plt.plot(freq, 20 * np.log10(spectrum / np.max(spectrum) + 1e-10))
    plt.title(f'{title} - Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(alpha=0.3)
    plt.xlim(0, 1000)  # Focus on sub-bass range
    
    plt.tight_layout()
    plt.show()

def save_sub_bass_examples(prefix="sub_bass"):
    """Generate and save different sub-bass examples"""
    # Basic sub-bass types
    basic_sine = generate_sub_bass(waveform="sine", duration=1.5)
    basic_triangle = generate_sub_bass(waveform="triangle", duration=1.5)
    basic_square = generate_sub_bass(waveform="square", saturation=0.2, duration=1.5)
    
    # Modulated sub-bass
    wobble_dubstep = create_wobble_bass(wobble_rate=2.0, format_type="dubstep", duration=3.0)
    wobble_neuro = create_wobble_bass(wobble_rate=4.0, format_type="neuro", duration=3.0)
    wobble_future = create_wobble_bass(wobble_rate=1.0, format_type="future", duration=3.0)
    
    # 808-style
    trap_808 = create_808_bass(format_type="trap", duration=2.0)
    modern_808 = create_808_bass(format_type="modern", duration=2.0)
    clean_808 = create_808_bass(format_type="clean", duration=2.0)
    
    # Reese bass
    reese_basic = create_reese_bass(num_oscillators=3, duration=2.0)
    reese_complex = create_reese_bass(num_oscillators=5, movement=0.4, duration=2.0)
    
    # Save all examples
    examples = {
        "sine": basic_sine,
        "triangle": basic_triangle,
        "square": basic_square,
        "wobble_dubstep": wobble_dubstep,
        "wobble_neuro": wobble_neuro,
        "wobble_future": wobble_future,
        "808_trap": trap_808,
        "808_modern": modern_808,
        "808_clean": clean_808,
        "reese_basic": reese_basic,
        "reese_complex": reese_complex
    }
    
    for name, wave in examples.items():
        filename = f"{prefix}_{name}.wav"
        wavfile.write(filename, 44100, (wave * 32767).astype(np.int16))
        print(f"Saved {filename}")
    
    return examples

# Example usage:
if __name__ == "__main__":
    # Generate and save examples
    examples = save_sub_bass_examples()
    
    # Visualize a couple of examples
    visualize_sub_bass(examples["wobble_dubstep"], title="Dubstep Wobble Bass")
    visualize_sub_bass(examples["808_modern"], title="Modern 808 Bass")
    visualize_sub_bass(examples["reese_complex"], title="Complex Reese Bass")
