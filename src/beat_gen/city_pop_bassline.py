import numpy as np
import random
from scipy.io import wavfile
from scipy import signal

class CityPopBasslineGenerator:
    """Generate sophisticated jazz-influenced city pop basslines"""
    
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
        
        # Extended jazz and modal scales
        self.scales = {
            # Major modes
            "C_ionian": ["C", "D", "E", "F", "G", "A", "B"],
            "C_dorian": ["C", "D", "D#", "F", "G", "A", "A#"],
            "C_phrygian": ["C", "C#", "D#", "F", "G", "G#", "A#"],
            "C_lydian": ["C", "D", "E", "F#", "G", "A", "B"],
            "C_mixolydian": ["C", "D", "E", "F", "G", "A", "A#"],
            "C_aeolian": ["C", "D", "D#", "F", "G", "G#", "A#"],
            "C_locrian": ["C", "C#", "D#", "F", "F#", "G#", "A#"],
            
            # Commonly used pentatonic scales
            "C_major_pentatonic": ["C", "D", "E", "G", "A"],
            "C_minor_pentatonic": ["C", "D#", "F", "G", "A#"],
            
            # Jazz harmony scales
            "C_melodic_minor": ["C", "D", "D#", "F", "G", "A", "B"],
            "C_harmonic_minor": ["C", "D", "D#", "F", "G", "G#", "B"],
            "C_harmonic_major": ["C", "D", "E", "F", "G", "G#", "B"],
            
            # City pop favorite scales
            "C_major7": ["C", "E", "G", "B"],
            "C_dominant9": ["C", "E", "G", "A#", "D"],
            "C_minor9": ["C", "D#", "G", "A#", "D"],
            "C_major13": ["C", "E", "G", "B", "D", "A"]
        }
        
        # Define extended chords
        self.chord_types = {
            "maj7": [0, 4, 7, 11],      # major 7th
            "7": [0, 4, 7, 10],         # dominant 7th
            "min7": [0, 3, 7, 10],      # minor 7th
            "min9": [0, 3, 7, 10, 14],  # minor 9th
            "maj9": [0, 4, 7, 11, 14],  # major 9th
            "9": [0, 4, 7, 10, 14],     # dominant 9th
            "min11": [0, 3, 7, 10, 14, 17],  # minor 11th
            "maj13": [0, 4, 7, 11, 14, 21],  # major 13th
            "13": [0, 4, 7, 10, 14, 21]      # dominant 13th
        }
        
        # Define common jazz/city-pop chord progressions
        self.progressions = {
            "city_pop_1": ["Imaj7", "IV7", "ii7", "V7"],
            "city_pop_2": ["Imaj7", "vi7", "ii7", "V7"],
            "city_pop_3": ["Imaj7", "IV7", "iii7", "bVII7"],
            "modal_1": ["Imaj7", "bVII7", "IV7", "bIII7"],
            "modal_2": ["Imaj7", "bVImaj7", "iimin7", "V9"],
            "deceptive_1": ["Imaj7", "iii7", "bIIImaj7", "bVImaj7"],
            "chromatic_1": ["Imaj7", "bIIImaj7", "V7", "bVII7"],
            "minor_jazz": ["imin9", "IV7", "bVImaj7", "V7alt"]
        }
        
        # City pop-style bassline patterns (emphasis on syncopation)
        self.patterns = {
            "smooth_1": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # Smooth syncopated
            "smooth_2": [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],  # More active
            "jazzy_1": [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],   # Jazz syncopation
            "fusion_1": [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],  # Fusion feel
            "walking": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],   # Walking bass
            "chromatic": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], # For chromatic walking
        }
    
    def create_modal_scale(self, root, mode):
        """Create a scale from any root note in any mode"""
        # Define semitones from root for each mode
        modes = {
            "ionian": [0, 2, 4, 5, 7, 9, 11],       # major scale
            "dorian": [0, 2, 3, 5, 7, 9, 10],
            "phrygian": [0, 1, 3, 5, 7, 8, 10],
            "lydian": [0, 2, 4, 6, 7, 9, 11],
            "mixolydian": [0, 2, 4, 5, 7, 9, 10],
            "aeolian": [0, 2, 3, 5, 7, 8, 10],      # natural minor
            "locrian": [0, 1, 3, 5, 6, 8, 10],
            "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
            "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
            "major_pentatonic": [0, 2, 4, 7, 9],
            "minor_pentatonic": [0, 3, 5, 7, 10]
        }
        
        # Get the semitones for the requested mode
        if mode not in modes:
            return ["C", "D", "E", "F", "G", "A", "B"]  # default to major
            
        semitones = modes[mode]
        
        # Get all notes in order
        all_notes = list(self.notes.keys())
        
        # Find the index of the root note
        root_idx = all_notes.index(root)
        
        # Build the scale
        scale = []
        for st in semitones:
            note_idx = (root_idx + st) % 12
            scale.append(all_notes[note_idx])
            
        return scale
    
    def create_chord(self, root, chord_type):
        """Create a chord from root note and chord type"""
        if chord_type not in self.chord_types:
            return [root]  # Default to just the root if chord type not found
        
        # Get all notes in order
        all_notes = list(self.notes.keys())
        
        # Find the index of the root note
        root_idx = all_notes.index(root)
        
        # Get semitones for this chord type
        semitones = self.chord_types[chord_type]
        
        # Build the chord
        chord = []
        for st in semitones:
            note_idx = (root_idx + st) % 12
            chord.append(all_notes[note_idx])
            
        return chord
    
    def roman_to_chord(self, roman, key="C"):
        """Convert Roman numeral notation to actual chord"""
        # Define mapping of scale degrees to indices
        roman_to_idx = {
            "I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5, "VII": 6,
            "i": 0, "ii": 1, "iii": 2, "iv": 3, "v": 4, "vi": 5, "vii": 6
        }
        
        # Handle flats
        if roman.startswith("b"):
            flat = True
            roman = roman[1:]  # Remove the flat symbol
        else:
            flat = False
        
        # Extract Roman numeral and chord type
        chord_type = ""
        base_roman = ""
        for i, char in enumerate(roman):
            if char.isalpha() and (char.isupper() or char.islower()):
                base_roman += char
            else:
                chord_type = roman[i:]
                break
        
        # Get the major scale for the key
        scale = self.create_modal_scale(key, "ionian")
        
        # Determine if minor or major based on capitalization
        is_minor = base_roman.islower()
        
        # Get index in the scale
        try:
            scale_idx = roman_to_idx[base_roman]
        except KeyError:
            return [key]  # Default to key note if invalid Roman numeral
        
        # Adjust for flat
        if flat:
            # Find the note at the scale index
            note_idx = list(self.notes.keys()).index(scale[scale_idx])
            # Flat the note
            note_idx = (note_idx - 1) % 12
            root = list(self.notes.keys())[note_idx]
        else:
            root = scale[scale_idx]
        
        # Set default chord type if not specified
        if not chord_type:
            if is_minor:
                chord_type = "min7"
            else:
                chord_type = "maj7"
                
        # Special case for V7alt (altered dominant)
        if chord_type == "7alt":
            chord_type = "7"  # For simplicity, treat as dominant 7
        
        # Create the chord
        return self.create_chord(root, chord_type)
    
    def generate_chord_progression(self, progression_type, key="C"):
        """Generate a chord progression from a predefined progression type"""
        if progression_type not in self.progressions:
            # Default to city_pop_1 if not found
            progression_type = "city_pop_1"
            
        roman_numerals = self.progressions[progression_type]
        
        progression = []
        for roman in roman_numerals:
            chord = self.roman_to_chord(roman, key)
            progression.append(chord)
            
        return progression
    
    def generate_walking_bassline(self, chord_progression, key="C", bars=4, complexity=0.7):
        """Generate a walking bassline from a chord progression"""
        sequence = []
        
        # Get scale degrees for the entire key to provide passing tones
        full_scale = self.create_modal_scale(key, "ionian")
        full_scale_indices = [list(self.notes.keys()).index(note) for note in full_scale]
        
        # For each chord in the progression
        for chord in chord_progression:
            # How many steps for this chord based on bars and progression length
            steps_per_chord = 16 // len(chord_progression)  # Assuming 16 steps per bar
            
            # Build a mini-sequence for this chord
            chord_sequence = []
            
            # Convert chord to indices for easier manipulation
            chord_indices = [list(self.notes.keys()).index(note) for note in chord]
            
            # Start with the root
            current_idx = chord_indices[0]
            chord_sequence.append((list(self.notes.keys())[current_idx], 2))  # Root, octave 2
            
            # Fill remaining steps with walking bass logic
            for i in range(1, steps_per_chord):
                # Decide what to do - chord tone or passing tone?
                if random.random() < 0.7:  # 70% chance of chord tone
                    if i == steps_per_chord - 1:  # Last note - lead to next chord
                        # Find next chord's root
                        next_chord_idx = (chord_progression.index(chord) + 1) % len(chord_progression)
                        next_root_idx = list(self.notes.keys()).index(chord_progression[next_chord_idx][0])
                        
                        # Find closest approach note
                        if next_root_idx > current_idx:
                            current_idx = next_root_idx - 1
                        elif next_root_idx < current_idx:
                            current_idx = next_root_idx + 1
                        else:
                            # If same note, pick a chord tone
                            current_idx = random.choice(chord_indices)
                    else:
                        # Normal chord tone
                        current_idx = random.choice(chord_indices)
                else:
                    # Use passing tone or approach tone logic
                    if i == steps_per_chord - 1:  # Lead to next chord
                        next_chord_idx = (chord_progression.index(chord) + 1) % len(chord_progression)
                        next_root_idx = list(self.notes.keys()).index(chord_progression[next_chord_idx][0])
                        
                        # Chromatic approach
                        if next_root_idx > current_idx:
                            current_idx = next_root_idx - 1
                        else:
                            current_idx = next_root_idx + 1
                    else:
                        # Prefer scale tones but allow chromatic passing tones
                        if random.random() < 0.3 * complexity:  # Chromatic passing tone
                            if random.random() < 0.5:
                                current_idx = (current_idx + 1) % 12
                            else:
                                current_idx = (current_idx - 1) % 12
                        else:  # Scale-based passing tone
                            # Find nearest scale tone that's not a chord tone
                            options = [idx for idx in full_scale_indices if idx not in chord_indices]
                            if options:
                                # Find closest option
                                distances = [abs(idx - current_idx) if abs(idx - current_idx) <= 6 
                                            else 12 - abs(idx - current_idx) for idx in options]
                                closest_idx = options[distances.index(min(distances))]
                                current_idx = closest_idx
                            else:
                                # If no options, just move by step
                                current_idx = (current_idx + 1) % 12
                
                # Ensure we're in a reasonable octave
                octave = 2
                if current_idx > 7:  # Higher notes get lower octave for bass
                    octave = 1
                
                # Add the note to the sequence
                chord_sequence.append((list(self.notes.keys())[current_idx], octave))
            
            # Add this chord's sequence to the main sequence
            sequence.extend(chord_sequence)
        
        # Make it the right length by truncating or repeating
        target_length = bars * 16
        if len(sequence) < target_length:
            # Repeat the sequence
            repetitions = target_length // len(sequence) + 1
            sequence = (sequence * repetitions)[:target_length]
        elif len(sequence) > target_length:
            # Truncate
            sequence = sequence[:target_length]
            
        return sequence
    
    def generate_city_pop_sequence(self, chord_progression, key="C", pattern_type="smooth_1", 
                                  bars=4, complexity=0.7):
        """Generate a city-pop style bassline from a chord progression"""
        # Get the pattern
        if pattern_type not in self.patterns:
            pattern_type = "smooth_1"
        
        pattern = self.patterns[pattern_type]
        pattern_length = len(pattern)
        
        # Initialize the sequence
        sequence = []
        
        # Number of steps in total
        total_steps = bars * 16
        
        # Number of steps per chord
        steps_per_chord = 16  # Simplifying to one chord per bar
        
        # For each bar
        for bar in range(bars):
            # Determine which chord to use
            chord_idx = bar % len(chord_progression)
            chord = chord_progression[chord_idx]
            
            # Convert chord to indices for easier manipulation
            all_notes = list(self.notes.keys())
            chord_indices = [all_notes.index(note) for note in chord]
            
            # Get scale for this key (for passing tones)
            scale = self.create_modal_scale(key, "ionian" if random.random() < 0.7 else "dorian")
            scale_indices = [all_notes.index(note) for note in scale]
            
            # Build a sequence for this bar
            bar_sequence = []
            
            for step in range(16):  # 16 steps per bar
                if pattern[step % pattern_length] == 1:
                    # This step has a note
                    
                    # Determine which note to play
                    if step == 0:
                        # First beat - use root note
                        note_idx = chord_indices[0]
                    elif step == 8 and random.random() < 0.6:
                        # Halfway through bar - often use the fifth
                        fifth_idx = -1
                        for idx in chord_indices:
                            if (idx - chord_indices[0]) % 12 == 7:  # Perfect fifth
                                fifth_idx = idx
                                break
                        if fifth_idx == -1:
                            fifth_idx = (chord_indices[0] + 7) % 12  # Fallback
                        note_idx = fifth_idx
                    elif random.random() < 0.75:
                        # Use a chord tone most of the time
                        note_idx = random.choice(chord_indices)
                    elif random.random() < 0.5:
                        # Use a scale tone sometimes
                        note_idx = random.choice(scale_indices)
                    else:
                        # Occasionally use a chromatic passing tone
                        # Find the last note we played
                        if bar_sequence and bar_sequence[-1] is not None:
                            last_note, _ = bar_sequence[-1]
                            last_idx = all_notes.index(last_note)
                            # Move chromatically
                            if random.random() < 0.5:
                                note_idx = (last_idx + 1) % 12  # Up
                            else:
                                note_idx = (last_idx - 1) % 12  # Down
                        else:
                            note_idx = chord_indices[0]
                    
                    # Determine octave (mostly 2, but can drop to 1 for variety)
                    if random.random() < 0.1 * complexity:
                        octave = 1
                    else:
                        octave = 2
                    
                    # Add note to sequence
                    bar_sequence.append((all_notes[note_idx], octave))
                else:
                    # This step is a rest
                    bar_sequence.append(None)
            
            # Add syncopated ghost notes based on complexity
            if complexity > 0.5:
                for step in range(16):
                    if bar_sequence[step] is None and random.random() < (complexity - 0.5) * 0.5:
                        # Add a "ghost note" - quieter transitional note
                        prev_step = step - 1
                        # Find the most recent non-None note
                        while prev_step >= 0 and bar_sequence[prev_step] is None:
                            prev_step -= 1
                            
                        if prev_step >= 0 and bar_sequence[prev_step] is not None:
                            # Continue from previous note
                            prev_note, prev_octave = bar_sequence[prev_step]
                            prev_idx = all_notes.index(prev_note)
                            
                            # Slightly modify the note for transition
                            note_idx = prev_idx
                            if random.random() < 0.7:
                                # Move by step (up or down)
                                direction = 1 if random.random() < 0.5 else -1
                                note_idx = (prev_idx + direction) % 12
                            
                            # Use the same octave as the previous note
                            bar_sequence[step] = (all_notes[note_idx], prev_octave)
            
            # Add this bar to the main sequence
            sequence.extend(bar_sequence)
        
        return sequence[:total_steps]  # Ensure correct length
    
    def calculate_note_frequency(self, note, octave):
        """Calculate the frequency of a note at a specific octave"""
        base_freq = self.notes.get(note, 49.00)  # Default to G if note not found
        return base_freq * (2 ** (octave - 1))
    
    def generate_city_pop_bass_sound(self, note_freq, duration, 
                                   roundness=0.7, brightness=0.5, harmonics=0.6):
        """Generate a smooth city-pop bass sound"""
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        # Base sine wave
        sine = np.sin(2 * np.pi * note_freq * t)
        
        # Triangle wave for roundness
        triangle = 2 * np.abs(2 * (t * note_freq - np.floor(t * note_freq + 0.5))) - 1
        
        # Square wave for brightness (slight)
        square = np.sign(np.sin(2 * np.pi * note_freq * t))
        
        # Combine waveforms based on roundness parameter
        waveform = sine * (1 - roundness) + triangle * roundness
        
        # Add brightness with a touch of square wave
        waveform = waveform * (1 - brightness*0.2) + square * (brightness*0.2)
        
        # Add harmonics
        if harmonics > 0:
            # First harmonic (octave)
            harmonic1 = np.sin(2 * np.pi * note_freq * 2 * t) * 0.3 * harmonics
            
            # Second harmonic (octave + fifth)
            harmonic2 = np.sin(2 * np.pi * note_freq * 3 * t) * 0.15 * harmonics
            
            # Add harmonics
            waveform += harmonic1 + harmonic2
        
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
        sound = waveform * env
        
        # Apply a subtle lowpass filter for warmth
        nyquist = self.sample_rate / 2
        cutoff = 1200  # Hz
        cutoff_normalized = cutoff / nyquist
        
        sos = signal.butter(4, cutoff_normalized, 'lowpass', output='sos')
        sound = signal.sosfilt(sos, sound)
        
        # Slight compression effect
        threshold = 0.5
        sound = np.where(
            np.abs(sound) > threshold,
            np.sign(sound) * (threshold + (np.abs(sound) - threshold) / 3),
            sound
        )
        
        return sound
    
    def render_city_pop_bassline(self, sequence, bpm, bass_params=None):
        """Render the city-pop bassline sequence to audio"""
        # Default bass parameters if none provided
        if bass_params is None:
            bass_params = {
                "roundness": 0.7,  # Blend of sine & triangle (0=pure sine, 1=pure triangle)
                "brightness": 0.5,  # Adds upper harmonics
                "harmonics": 0.6    # Amount of additional harmonic content
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
                sound = self.generate_city_pop_bass_sound(
                    note_freq, 
                    note_duration * 1.2,  # Slightly longer to allow for release
                    roundness=bass_params.get("roundness", 0.7),
                    brightness=bass_params.get("brightness", 0.5),
                    harmonics=bass_params.get("harmonics", 0.6)
                )
                
                # Add to output
                end_pos = min(position + len(sound), total_samples)
                bassline_audio[position:end_pos] += sound[:end_pos - position]
        
        # Normalize
        if np.max(np.abs(bassline_audio)) > 0:
            bassline_audio = bassline_audio / np.max(np.abs(bassline_audio)) * 0.9
        
        return bassline_audio
    
    def generate_city_pop_bassline(self, key="C", chord_progression_type="city_pop_1", 
                                 pattern_type="smooth_1", bars=4, complexity=0.7,
                                 walking_bass=False, bass_params=None):
        """Generate a complete city-pop style bassline"""
        # Generate chord progression
        chord_progression = self.generate_chord_progression(chord_progression_type, key)
        
        # Generate the bassline sequence
        if walking_bass:
            sequence = self.generate_walking_bassline(chord_progression, key, bars, complexity)
        else:
            sequence = self.generate_city_pop_sequence(
                chord_progression, key, pattern_type, bars, complexity
            )
        
        # Render the bassline to audio
        bpm = 95  # City pop tends to be mid-tempo
        bassline_audio = self.render_city_pop_bassline(sequence, bpm, bass_params)
        
        return bassline_audio, sequence, chord_progression
    
    def save_bassline(self, bassline_audio, filename="city_pop_bassline.wav"):
        """Save the bassline as a WAV file"""
        wavfile.write(filename, self.sample_rate, (bassline_audio * 32767).astype(np.int16))
        print(f"City-pop bassline saved to {filename}")

# For testing
if __name__ == "__main__":
    import random
    
    # Create the generator
    generator = CityPopBasslineGenerator()
    
    # Generate a city pop bassline
    bassline, sequence, chords = generator.generate_city_pop_bassline(
        key="F",
        chord_progression_type="city_pop_2",
        pattern_type="smooth_2",
        bars=4,
        complexity=0.7,
        walking_bass=False
    )
    
    # Save the bassline
    generator.save_bassline(bassline, "city_pop_bassline.wav")
    
    # Print chord progression for reference
    print("Chord Progression:")
    for chord in chords:
        print(chord)
