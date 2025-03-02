import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import matplotlib.pyplot as plt
import argparse

# Import beat-gen components
from beat_gen.edm_drum_library import get_kit_by_name
from beat_gen.edm_beat_generator import (
    BeatGenerator,
    generate_house_beat,
    generate_techno_beat,
    generate_dubstep_beat,
    generate_hardstyle_beat
)

class BasslineGenerator:
    """Generate basslines that complement generated beats"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.notes = {
            "C": 32.70,  # C1
            "C#": 34.65,
            "D": 36.71,
            "D#": 38.89,
            "E": 41.20,
            "F": 43.65,
            "F#": 46.25,
            "G": 49.00,
            "G#": 51.91,
            "A": 55.00,
            "A#": 58.27,
            "B": 61.74
        }
        # Common EDM keys
        self.scales = {
            "C_minor": ["C", "D", "D#", "F", "G", "G#", "A#"],
            "F_minor": ["F", "G", "G#", "A#", "C", "C#", "D#"],
            "G_minor": ["G", "A", "A#", "C", "D", "D#", "F"],
            "A_minor": ["A", "B", "C", "D", "E", "F", "G"],
            "D_minor": ["D", "E", "F", "G", "A", "A#", "C"],
            "E_minor": ["E", "F#", "G", "A", "B", "C", "D"]
        }
        # Define some common bassline patterns for different genres
        self.patterns = {
            "house": [
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Basic pulse
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # Syncopated
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Four-on-floor
                [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]   # More movement
            ],
            "techno": [
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Steady
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Minimal
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Driving
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1]   # Accent pattern
            ],
            "dubstep": [
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Half-time
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Sparse
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Wobble rhythm
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]   # Complex
            ],
            "hardstyle": [
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Straight four
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],  # Kick roll
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],  # Accented
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]   # Driving
            ]
        }
        
    def detect_key_from_kick(self, beat, beat_generator):
        """
        Analyze the kick frequency to suggest a compatible bassline key
        """
        # For simplicity, we'll just return a predefined key that works well with EDM
        # In a more advanced implementation, this would analyze the kick frequency
        
        # Get kick sound info from the generator
        kick_sounds = beat_generator.drum_kit.get_sounds_by_category("kick")
        if kick_sounds:
            # For now, just use a simple mapping based on the genre
            if "house" in beat_generator.drum_kit.name.lower():
                return "F_minor"
            elif "techno" in beat_generator.drum_kit.name.lower():
                return "G_minor"
            elif "dubstep" in beat_generator.drum_kit.name.lower():
                return "C_minor"
            elif "hardstyle" in beat_generator.drum_kit.name.lower():
                return "A_minor"
        
        # Default to C minor (common in EDM)
        return "C_minor"
    
    def detect_pattern_from_beat(self, beat, beat_generator):
        """
        Analyze the beat to determine the most appropriate bassline pattern
        """
        # Get the genre from the beat generator
        genre = "house"  # Default
        
        if "house" in beat_generator.drum_kit.name.lower():
            genre = "house"
        elif "techno" in beat_generator.drum_kit.name.lower():
            genre = "techno"
        elif "dubstep" in beat_generator.drum_kit.name.lower():
            genre = "dubstep"
        elif "hardstyle" in beat_generator.drum_kit.name.lower():
            genre = "hardstyle"
        
        # Choose a pattern based on genre and randomly select from options
        available_patterns = self.patterns.get(genre, self.patterns["house"])
        pattern_index = np.random.randint(0, len(available_patterns))
        return available_patterns[pattern_index]
    
    def generate_bassline_sequence(self, key, pattern, complexity=0.5, bars=4, notes_per_bar=16):
        """
        Generate a sequence of notes based on the key and pattern
        """
        # Get the scale for the given key
        scale = self.scales.get(key, self.scales["C_minor"])
        
        # Create note sequence
        sequence = []
        
        # Base pattern is one bar long, repeat for the requested number of bars
        for bar in range(bars):
            # Choose whether to use pattern variation for this bar
            use_variation = (np.random.random() < complexity * 0.4) and bar > 0
            
            bar_sequence = []
            for i in range(notes_per_bar):
                if pattern[i % len(pattern)] == 1:
                    # For the first beat of the bar, prefer the root note
                    if i == 0 and np.random.random() < 0.8:
                        note = scale[0]  # Root note
                    else:
                        # Choose a note from the scale
                        # With higher complexity, use more varied notes
                        max_note_index = min(3 + int(complexity * 4), len(scale) - 1)
                        note_index = np.random.randint(0, max_note_index + 1)
                        note = scale[note_index]
                    
                    # Determine octave (1-3) - basslines usually use lower octaves
                    octave = np.random.randint(1, 3)
                    if complexity < 0.3:
                        octave = 1  # Simpler basslines stay in lower octave
                    
                    # Add note to sequence
                    bar_sequence.append((note, octave))
                else:
                    # Rest
                    bar_sequence.append(None)
            
            # Add some variations if needed
            if use_variation:
                # Add occasional note for interest
                for i in range(notes_per_bar):
                    if bar_sequence[i] is None and np.random.random() < complexity * 0.2:
                        note_index = np.random.randint(0, len(scale))
                        note = scale[note_index]
                        octave = np.random.randint(1, 3)
                        bar_sequence[i] = (note, octave)
            
            sequence.extend(bar_sequence)
        
        return sequence
    
    def calculate_note_frequency(self, note, octave):
        """
        Calculate the frequency of a note at a specific octave
        """
        base_freq = self.notes.get(note, 49.00)  # Default to G if note not found
        return base_freq * (2 ** (octave - 1))
    
    def generate_bass_sound(self, note_freq, duration, waveform="saw", 
                            filter_cutoff=1000, resonance=1.0, distortion=0.0):
        """
        Generate a bass sound with the given parameters
        """
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        # Generate base waveform
        if waveform == "sine":
            sound = np.sin(2 * np.pi * note_freq * t)
        elif waveform == "triangle":
            sound = 2 * np.abs(2 * (t * note_freq - np.floor(t * note_freq + 0.5))) - 1
        elif waveform == "square":
            sound = np.sign(np.sin(2 * np.pi * note_freq * t))
        elif waveform == "saw":
            sound = 2 * (t * note_freq - np.floor(t * note_freq)) - 1
        else:
            # Default to saw
            sound = 2 * (t * note_freq - np.floor(t * note_freq)) - 1
        
        # Add a sub-oscillator for more bass (one octave down)
        sub_osc = np.sin(2 * np.pi * note_freq / 2 * t) * 0.5
        sound = sound * 0.7 + sub_osc * 0.3
        
        # Apply amplitude envelope (ADSR)
        attack = 0.01  # 10ms attack
        decay = 0.1    # 100ms decay
        sustain = 0.7  # 70% sustain level
        release = 0.1  # 100ms release
        
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        
        env = np.ones(num_samples)
        
        # Attack phase
        if attack_samples > 0:
            env[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        if decay_samples > 0:
            decay_end = min(attack_samples + decay_samples, num_samples)
            decay_range = np.linspace(1, sustain, decay_samples)
            env[attack_samples:decay_end] = decay_range[:decay_end-attack_samples]
        
        # Release phase
        if release_samples > 0 and num_samples > release_samples:
            env[-release_samples:] = np.linspace(env[-(release_samples+1)], 0, release_samples)
        
        # Apply envelope
        sound = sound * env
        
        # Apply lowpass filter
        if filter_cutoff > 0:
            # Increase resonance by narrowing the filter
            nyquist = self.sample_rate / 2
            cutoff_normalized = filter_cutoff / nyquist
            
            # Use higher order for more resonance
            filter_order = 4
            if resonance > 1.0:
                filter_order = int(4 + resonance * 4)
            
            sos = signal.butter(filter_order, cutoff_normalized, 'lowpass', output='sos')
            sound = signal.sosfilt(sos, sound)
        
        # Apply distortion if needed
        if distortion > 0:
            sound = np.tanh(sound * (1 + distortion * 5)) / (1 + distortion)
        
        return sound
    
    def render_bassline(self, sequence, bpm, bass_params=None):
        """
        Render the bassline sequence to audio
        """
        # Default bass parameters if none provided
        if bass_params is None:
            bass_params = {
                "waveform": "saw",
                "filter_cutoff": 800,
                "resonance": 1.2,
                "distortion": 0.2
            }
        
        # Calculate note duration based on BPM (assuming 16th notes)
        beat_duration = 60 / bpm  # Duration of one beat in seconds
        note_duration = beat_duration / 4  # 16th notes (4 per beat)
        
        # Create empty audio data
        total_samples = int(len(sequence) * note_duration * self.sample_rate)
        bassline_audio = np.zeros(total_samples)
        
        # Render each note in the sequence
        for i, note_info in enumerate(sequence):
            if note_info is not None:
                note, octave = note_info
                note_freq = self.calculate_note_frequency(note, octave)
                
                # Calculate position
                position = int(i * note_duration * self.sample_rate)
                
                # Generate bass sound
                sound = self.generate_bass_sound(
                    note_freq, 
                    note_duration * 1.2,  # Slightly longer to allow for release
                    waveform=bass_params.get("waveform", "saw"),
                    filter_cutoff=bass_params.get("filter_cutoff", 800),
                    resonance=bass_params.get("resonance", 1.2),
                    distortion=bass_params.get("distortion", 0.2)
                )
                
                # Add to output
                end_pos = min(position + len(sound), total_samples)
                bassline_audio[position:end_pos] += sound[:end_pos - position]
        
        # Normalize
        if np.max(np.abs(bassline_audio)) > 0:
            bassline_audio = bassline_audio / np.max(np.abs(bassline_audio)) * 0.9
        
        return bassline_audio
    
    def generate_bassline_for_beat(self, beat, beat_generator, complexity=0.5, bass_params=None):
        """
        Generate a bassline that complements the given beat
        """
        # Detect key and pattern from beat
        key = self.detect_key_from_kick(beat, beat_generator)
        pattern = self.detect_pattern_from_beat(beat, beat_generator)
        
        # Calculate the number of bars in the beat
        beat_bars = len(beat) // beat_generator.bar_length
        
        # Generate bassline sequence
        sequence = self.generate_bassline_sequence(
            key=key,
            pattern=pattern,
            complexity=complexity,
            bars=beat_bars
        )
        
        # Render bassline
        bassline_audio = self.render_bassline(
            sequence=sequence,
            bpm=beat_generator.bpm,
            bass_params=bass_params
        )
        
        # Make sure bassline matches beat length
        if len(bassline_audio) > len(beat):
            bassline_audio = bassline_audio[:len(beat)]
        elif len(bassline_audio) < len(beat):
            padding = np.zeros(len(beat) - len(bassline_audio))
            bassline_audio = np.concatenate([bassline_audio, padding])
        
        return bassline_audio, sequence, key
    
    def save_bassline(self, bassline_audio, filename="bassline.wav"):
        """Save the bassline as a WAV file"""
        wavfile.write(filename, self.sample_rate, (bassline_audio * 32767).astype(np.int16))
        print(f"Bassline saved to {filename}")
    
    def visualize_bassline(self, bassline_audio, sequence, key, bpm):
        """Visualize the bassline waveform and note pattern"""
        plt.figure(figsize=(12, 8))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(bassline_audio)) / self.sample_rate, bassline_audio)
        plt.title(f'Bassline Waveform (Key: {key}, BPM: {bpm})')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(alpha=0.3)
        
        # Plot spectrogram
        plt.subplot(2, 1, 2)
        plt.specgram(bassline_audio, NFFT=1024, Fs=self.sample_rate, noverlap=512, cmap='inferno')
        plt.title('Bassline Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Intensity (dB)')
        plt.ylim(0, 2000)  # Focus on bass frequencies
        
        plt.tight_layout()
        plt.show()


def generate_beat_with_bassline(
    genre="house", 
    bpm=128, 
    bars=4, 
    complexity=0.5, 
    swing=0.0,
    bass_complexity=0.5,
    bass_waveform="saw",
    output_dir="output",
    visualize=False
):
    """
    Generate a beat with a complementary bassline
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate beat based on genre
    print(f"Generating {genre} beat...")
    
    if genre == 'house':
        beat, generator = generate_house_beat(bpm=bpm, bars=bars, complexity=complexity, swing=swing)
    elif genre == 'techno':
        beat, generator = generate_techno_beat(bpm=bpm, bars=bars, complexity=complexity, swing=swing)
    elif genre == 'dubstep':
        beat, generator = generate_dubstep_beat(bpm=bpm, bars=bars, complexity=complexity, swing=swing)
    elif genre == 'hardstyle':
        beat, generator = generate_hardstyle_beat(bpm=bpm, bars=bars, complexity=complexity, swing=swing)
    else:
        # Fallback to house
        beat, generator = generate_house_beat(bpm=bpm, bars=bars, complexity=complexity, swing=swing)
    
    # Generate bassline
    print("Generating matching bassline...")
    bassline_generator = BasslineGenerator(sample_rate=generator.sample_rate)
    
    # Configure bass parameters based on genre
    bass_params = {
        "waveform": bass_waveform,
        "filter_cutoff": 800,
        "resonance": 1.2,
        "distortion": 0.2
    }
    
    # Adjust parameters based on genre
    if genre == 'house':
        bass_params["filter_cutoff"] = 900
        bass_params["resonance"] = 1.5
    elif genre == 'techno':
        bass_params["filter_cutoff"] = 600
        bass_params["resonance"] = 2.0
        bass_params["distortion"] = 0.3
    elif genre == 'dubstep':
        bass_params["filter_cutoff"] = 400
        bass_params["resonance"] = 3.0
        bass_params["distortion"] = 0.5
    elif genre == 'hardstyle':
        bass_params["filter_cutoff"] = 700
        bass_params["resonance"] = 2.5
        bass_params["distortion"] = 0.4
    
    # Generate bassline for the beat
    bassline, sequence, key = bassline_generator.generate_bassline_for_beat(
        beat, generator, complexity=bass_complexity, bass_params=bass_params
    )
    
    # Mix beat and bassline
    mixed = beat * 0.7 + bassline * 0.7
    
    # Normalize the final mix
    mixed = mixed / np.max(np.abs(mixed)) * 0.95
    
    # Save individual components and the mixed track
    beat_filename = os.path.join(output_dir, f"{genre}_beat_{bpm}bpm.wav")
    bassline_filename = os.path.join(output_dir, f"{genre}_bassline_{bpm}bpm.wav")
    mixed_filename = os.path.join(output_dir, f"{genre}_beat_with_bassline_{bpm}bpm.wav")
    
    generator.save_beat(beat, beat_filename)
    bassline_generator.save_bassline(bassline, bassline_filename)
    wavfile.write(mixed_filename, generator.sample_rate, (mixed * 32767).astype(np.int16))
    
    print(f"Mixed track saved to: {mixed_filename}")
    
    # Visualize if requested
    if visualize:
        print("Generating visualizations...")
        generator.visualize_beat(beat)
        bassline_generator.visualize_bassline(bassline, sequence, key, bpm)


def main():
    """Main function to parse arguments and generate beat with bassline"""
    parser = argparse.ArgumentParser(
        description="Generate EDM beats with complementary basslines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic parameters
    parser.add_argument("--genre", type=str, default="house", 
                        choices=["house", "techno", "dubstep", "hardstyle"],
                        help="Genre of beat to generate")
    parser.add_argument("--bpm", type=int, default=128,
                        help="Beats per minute")
    parser.add_argument("--bars", type=int, default=4,
                        help="Number of bars to generate")
    
    # Beat parameters
    parser.add_argument("--complexity", type=float, default=0.5,
                        help="Complexity of the beat (0.0-1.0)")
    parser.add_argument("--swing", type=float, default=0.0,
                        help="Amount of swing (0.0-1.0)")
    
    # Bassline parameters
    parser.add_argument("--bass-complexity", type=float, default=0.5,
                        help="Complexity of the bassline (0.0-1.0)")
    parser.add_argument("--bass-waveform", type=str, default="saw",
                        choices=["saw", "square", "triangle", "sine"],
                        help="Waveform type for the bassline")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate waveform and spectrogram visualization")
    
    args = parser.parse_args()
    
    generate_beat_with_bassline(
        genre=args.genre,
        bpm=args.bpm,
        bars=args.bars,
        complexity=args.complexity,
        swing=args.swing,
        bass_complexity=args.bass_complexity,
        bass_waveform=args.bass_waveform,
        output_dir=args.output_dir,
        visualize=args.visualize
    )

if __name__ == "__main__":
    main()
