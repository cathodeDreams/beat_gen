import numpy as np
import random
from scipy.io import wavfile
import matplotlib.pyplot as plt

from beat_gen.edm_drum_library import (get_kit_by_name, create_kick, create_snare, 
                            create_hat, create_percussion, create_sub_bass, DrumSound)

class BeatGenerator:
    """Class for generating EDM beats"""
    
    def __init__(self, drum_kit, bpm=128, sample_rate=44100):
        self.drum_kit = drum_kit
        self.bpm = bpm
        self.sample_rate = sample_rate
        self.beat_length = int(60 / bpm * sample_rate)  # Length of one beat in samples
        self.bar_length = self.beat_length * 4  # 4 beats per bar in 4/4 time
        
        # Common pattern lengths
        self.pattern_lengths = {
            "kick": 4,       # One bar pattern (quarter notes)
            "snare": 4,      # One bar pattern (quarter notes)
            "hat": 16,       # 16th note pattern (four 16th notes per beat)
            "perc": 8,       # 8th note pattern (two 8th notes per beat)
            "open_hat": 8,   # 8th note pattern for open hats
            "sub_bass": 4    # One bar pattern (quarter notes) for sub-bass
        }
    
    def generate_pattern(self, category, pattern_length, density=0.5, variation=0.2):
        """Generate a binary pattern (0 or 1) for a drum category"""
        # Start with an empty pattern
        pattern = [0] * pattern_length
        
        if category == "kick":
            # 4-on-the-floor: kick on every beat
            for i in range(0, pattern_length, pattern_length // 4):
                pattern[i] = 1
            
            # Add occasional off-beat kicks based on variation
            for i in range(1, pattern_length, 2):
                if random.random() < variation * 0.3:
                    pattern[i] = 1
        
        elif category == "snare":
            # Standard snare on beats 2 and 4
            for i in range(pattern_length // 4, pattern_length, pattern_length // 2):
                pattern[i] = 1
            
            # Add occasional off-beat snares based on variation
            snare_variation_points = [i for i in range(pattern_length) 
                                    if i % (pattern_length // 4) != 0]
            for i in snare_variation_points:
                if random.random() < variation * 0.2:
                    pattern[i] = 1
        
        elif category == "hat":
            # Hats often play on every 8th or 16th note
            hat_interval = 1 if density > 0.7 else 2
            
            for i in range(0, pattern_length, hat_interval):
                if random.random() < density:
                    pattern[i] = 1
            
            # Ensure we have hats on key points if density is high enough
            if density > 0.4:
                for i in range(0, pattern_length, 4):  # Every quarter note
                    if random.random() < 0.9:  # 90% chance
                        pattern[i] = 1
        
        elif category == "open_hat":
            # Open hats are less frequent, often on off-beats
            for i in range(1, pattern_length, 2):  # Off-beats
                if random.random() < density * 0.3:
                    pattern[i] = 1
        
        elif category == "perc":
            # Percussion typically adds accents on certain beats
            # More active on the second half of the bar
            for i in range(pattern_length):
                if i >= pattern_length / 2:
                    # More active in second half
                    if random.random() < density * 0.7:
                        pattern[i] = 1
                else:
                    # Less active in first half
                    if random.random() < density * 0.3:
                        pattern[i] = 1
        
        elif category == "sub_bass":
            # Determine style based on density and variation
            if density > 0.7:  # Busy, complex bassline
                # Add bass on downbeats and some offbeats
                for i in range(0, pattern_length, 2):
                    if i % 4 == 0 or random.random() < variation * 0.8:
                        pattern[i] = 1
            elif density > 0.4:  # Medium activity
                # Bass on first beat and occasionally on third
                pattern[0] = 1
                if random.random() < variation * 0.7:
                    pattern[pattern_length // 2] = 1
                
                # Add occasional off-beat notes
                for i in range(1, pattern_length, 2):
                    if random.random() < variation * 0.3:
                        pattern[i] = 1
            else:  # Simple, minimal bassline
                # Just the downbeat
                pattern[0] = 1
                
                # Maybe add more notes based on variation
                if variation > 0.4 and random.random() < 0.3:
                    offset = random.choice([pattern_length // 4 * 2, pattern_length // 4 * 3])
                    pattern[offset] = 1
        
        return pattern
    
    def get_velocity_pattern(self, pattern, humanize=0.2):
        """Generate velocity values (0.0-1.0) for each hit in a pattern"""
        velocities = []
        
        for i, hit in enumerate(pattern):
            if hit == 1:
                # Base velocity depending on position in bar
                if i % (len(pattern) // 4) == 0:
                    # Emphasize downbeats
                    base_velocity = 1.0
                elif i % 2 == 0:
                    # Medium velocity for even numbered steps
                    base_velocity = 0.85
                else:
                    # Lower velocity for odd numbered steps
                    base_velocity = 0.7
                
                # Add humanization
                velocity = base_velocity * (1 - humanize/2 + random.random() * humanize)
                velocities.append(max(0.1, min(1.0, velocity)))
            else:
                velocities.append(0)
        
        return velocities
    
    def render_pattern(self, pattern, velocities, sound, steps_per_beat):
        """Render a pattern to audio using a specific sound"""
        # Calculate total length in samples
        total_steps = len(pattern)
        samples_per_step = self.beat_length // steps_per_beat
        total_samples = samples_per_step * total_steps
        
        # Create empty audio data
        audio_data = np.zeros(total_samples)
        
        # Render each hit
        for i, (hit, velocity) in enumerate(zip(pattern, velocities)):
            if hit == 1:
                # Calculate position
                position = i * samples_per_step
                
                # Get sound data
                sound_data = sound.audio_data * velocity
                
                # Add to output
                end_pos = min(position + len(sound_data), total_samples)
                audio_data[position:end_pos] += sound_data[:end_pos - position]
        
        return audio_data
    
    def generate_fill(self, length=1, intensity=0.5):
        """Generate a drum fill to add variation"""
        # Length is in beats (quarter notes)
        fill_samples = self.beat_length * length
        fill_audio = np.zeros(fill_samples)
        
        # Get snare for the fill
        snare_sounds = self.drum_kit.get_sounds_by_category("snare")
        if not snare_sounds:
            return fill_audio
        
        snare = snare_sounds[0]
        
        # Determine number of hits based on intensity
        num_hits = int(2 + intensity * 10)  # 2 to 12 hits
        
        # Place hits with increasing density toward the end
        for i in range(num_hits):
            # More hits toward the end of the fill
            position_factor = i / num_hits
            position = int(fill_samples * (0.2 + 0.8 * position_factor))
            
            # Add some randomization to position
            jitter = int(self.beat_length * 0.1)  # 10% of a beat
            position += random.randint(-jitter, jitter)
            position = max(0, min(position, fill_samples - 1))
            
            # Increasingly louder toward the end
            velocity = 0.5 + 0.5 * position_factor
            
            # Add the hit
            sound_data = snare.audio_data * velocity
            end_pos = min(position + len(sound_data), fill_samples)
            fill_audio[position:end_pos] += sound_data[:end_pos - position]
        
        return fill_audio
    
    def apply_swing(self, pattern, swing_amount=0.3):
        """Apply swing feel to a pattern"""
        # Only makes sense for patterns with even divisions
        if len(pattern) % 2 != 0:
            return pattern
        
        swung_pattern = pattern.copy()
        
        # For every pair of notes, adjust the timing of the second note
        for i in range(1, len(pattern), 2):
            if pattern[i] == 1:
                # Remove the hit from its current position
                swung_pattern[i] = 0
                
                # Place it at a slightly later position
                # In a real sequencer this would be a timing offset,
                # but for this simplified version we're just moving it to the next step
                if random.random() < swing_amount and i+1 < len(pattern):
                    swung_pattern[i+1] = 1
                else:
                    # Keep it at the original position if not swung
                    swung_pattern[i] = 1
        
        return swung_pattern
    
    def generate_beat(self, bars=4, swing=0.0, complexity=0.5, humanize=0.2):
        """Generate a complete beat with multiple drum sounds"""
        # Calculate total length
        total_samples = self.bar_length * bars
        
        # Create empty output
        output = np.zeros(total_samples)
        
        # Generate patterns for each drum category
        
        # Kick pattern (4 steps per bar, 1 bar long)
        kick_pattern_length = self.pattern_lengths["kick"]
        kick_pattern = self.generate_pattern("kick", kick_pattern_length, 
                                           density=0.9, variation=complexity * 0.5)
        kick_velocities = self.get_velocity_pattern(kick_pattern, humanize)
        
        # Get kick sound
        kick_sound = self.drum_kit.get_sounds_by_category("kick")[0]
        
        # Calculate steps per beat for kick
        kick_steps_per_beat = kick_pattern_length // 4
        
        # Render kick pattern for all bars
        for bar in range(bars):
            # Add variation to last bar if it's a multiple of 4
            if bar > 0 and (bar + 1) % 4 == 0 and random.random() < complexity * 0.7:
                # Create a fill for the last beat of the bar
                fill_position = bar * self.bar_length + self.beat_length * 3
                fill_audio = self.generate_fill(1, complexity)
                
                # Add fill to output
                end_pos = min(fill_position + len(fill_audio), total_samples)
                output[fill_position:end_pos] += fill_audio[:end_pos - fill_position]
                
                # Render the first 3 beats of the kick pattern
                kick_audio_partial = self.render_pattern(kick_pattern[:3], kick_velocities[:3], 
                                                      kick_sound, kick_steps_per_beat)
                bar_position = bar * self.bar_length
                output[bar_position:bar_position + len(kick_audio_partial)] += kick_audio_partial
            else:
                # Render full kick pattern
                kick_audio = self.render_pattern(kick_pattern, kick_velocities, kick_sound, kick_steps_per_beat)
                bar_position = bar * self.bar_length
                output[bar_position:bar_position + len(kick_audio)] += kick_audio
        
        # Snare pattern (4 steps per bar, 1 bar long)
        snare_pattern_length = self.pattern_lengths["snare"]
        snare_pattern = self.generate_pattern("snare", snare_pattern_length, 
                                            density=0.5, variation=complexity * 0.7)
        snare_velocities = self.get_velocity_pattern(snare_pattern, humanize)
        
        # Get snare sound
        snare_sounds = self.drum_kit.get_sounds_by_category("snare")
        snare_sound = snare_sounds[0] if snare_sounds else None
        
        if snare_sound:
            # Calculate steps per beat for snare
            snare_steps_per_beat = snare_pattern_length // 4
            
            # Render snare pattern for all bars
            for bar in range(bars):
                # Skip the last beat if we have a fill
                if bar > 0 and (bar + 1) % 4 == 0 and random.random() < complexity * 0.7:
                    # Render the first 3 beats of the snare pattern
                    snare_audio_partial = self.render_pattern(snare_pattern[:3], snare_velocities[:3], 
                                                         snare_sound, snare_steps_per_beat)
                    bar_position = bar * self.bar_length
                    output[bar_position:bar_position + len(snare_audio_partial)] += snare_audio_partial
                else:
                    # Render full snare pattern
                    snare_audio = self.render_pattern(snare_pattern, snare_velocities, 
                                                 snare_sound, snare_steps_per_beat)
                    bar_position = bar * self.bar_length
                    output[bar_position:bar_position + len(snare_audio)] += snare_audio
        
        # Hat pattern (16 steps per bar, 1 bar long)
        hat_pattern_length = self.pattern_lengths["hat"]
        hat_pattern = self.generate_pattern("hat", hat_pattern_length, 
                                          density=0.7 + complexity * 0.3, 
                                          variation=complexity * 0.5)
        
        # Apply swing to hat pattern if requested
        if swing > 0:
            hat_pattern = self.apply_swing(hat_pattern, swing)
        
        hat_velocities = self.get_velocity_pattern(hat_pattern, humanize)
        
        # Get hat sounds
        hat_sounds = self.drum_kit.get_sounds_by_category("hat")
        closed_hat = next((s for s in hat_sounds if "Closed" in s.name), None)
        open_hat = next((s for s in hat_sounds if "Open" in s.name), None)
        
        if closed_hat:
            # Calculate steps per beat for hat
            hat_steps_per_beat = hat_pattern_length // 4
            
            # Render closed hat pattern for all bars
            for bar in range(bars):
                bar_position = bar * self.bar_length
                hat_audio = self.render_pattern(hat_pattern, hat_velocities, 
                                             closed_hat, hat_steps_per_beat)
                output[bar_position:bar_position + len(hat_audio)] += hat_audio
        
        # Add open hats for variety
        if open_hat and complexity > 0.3:
            # Generate open hat pattern - sparser than closed hats
            open_hat_pattern_length = self.pattern_lengths["open_hat"]
            open_hat_pattern = self.generate_pattern("open_hat", open_hat_pattern_length, 
                                                  density=complexity * 0.4, 
                                                  variation=complexity)
            
            # Apply swing if needed
            if swing > 0:
                open_hat_pattern = self.apply_swing(open_hat_pattern, swing)
                
            open_hat_velocities = self.get_velocity_pattern(open_hat_pattern, humanize)
            
            # Calculate steps per beat for open hat
            open_hat_steps_per_beat = open_hat_pattern_length // 4
            
            # Render open hat pattern for all bars
            for bar in range(bars):
                bar_position = bar * self.bar_length
                open_hat_audio = self.render_pattern(open_hat_pattern, open_hat_velocities, 
                                                  open_hat, open_hat_steps_per_beat)
                output[bar_position:bar_position + len(open_hat_audio)] += open_hat_audio
        
        # Add percussion if complexity is high enough
        if complexity > 0.3:
            perc_sounds = self.drum_kit.get_sounds_by_category("percussion")
            
            if perc_sounds:
                # Use first percussion sound
                perc_sound = perc_sounds[0]
                
                # Generate percussion pattern
                perc_pattern_length = self.pattern_lengths["perc"]
                perc_pattern = self.generate_pattern("perc", perc_pattern_length, 
                                                   density=complexity * 0.4, 
                                                   variation=complexity)
                
                # Apply swing if needed
                if swing > 0:
                    perc_pattern = self.apply_swing(perc_pattern, swing)
                    
                perc_velocities = self.get_velocity_pattern(perc_pattern, humanize * 1.5)
                
                # Calculate steps per beat for percussion
                perc_steps_per_beat = perc_pattern_length // 4
                
                # Render percussion pattern for all bars
                for bar in range(bars):
                    bar_position = bar * self.bar_length
                    perc_audio = self.render_pattern(perc_pattern, perc_velocities, 
                                                  perc_sound, perc_steps_per_beat)
                    output[bar_position:bar_position + len(perc_audio)] += perc_audio
        
        # Add second percussion sound if available and complexity is high
        if complexity > 0.6:
            perc_sounds = self.drum_kit.get_sounds_by_category("percussion")
            
            if len(perc_sounds) > 1:
                # Use second percussion sound
                perc2_sound = perc_sounds[1]
                
                # Generate a different percussion pattern
                perc2_pattern_length = self.pattern_lengths["perc"] * 2  # Longer pattern
                perc2_pattern = self.generate_pattern("perc", perc2_pattern_length, 
                                                    density=complexity * 0.3, 
                                                    variation=complexity * 1.2)
                
                # Apply swing if needed
                if swing > 0:
                    perc2_pattern = self.apply_swing(perc2_pattern, swing)
                    
                perc2_velocities = self.get_velocity_pattern(perc2_pattern, humanize * 2)
                
                # Calculate steps per beat
                perc2_steps_per_beat = perc2_pattern_length // 8  # 2 bars
                
                # Render this percussion pattern with a two-bar cycle
                for bar in range(0, bars, 2):
                    if bar >= bars:
                        break
                    bar_position = bar * self.bar_length
                    perc2_audio = self.render_pattern(perc2_pattern, perc2_velocities, 
                                                   perc2_sound, perc2_steps_per_beat)
                    end_pos = min(bar_position + len(perc2_audio), total_samples)
                    output[bar_position:end_pos] += perc2_audio[:end_pos - bar_position]
        
        # Add sub-bass if available and complexity is high enough
        if complexity > 0.3:
            sub_bass = self.drum_kit.get_sub_bass()
            
            if sub_bass:
                # Determine style based on kit name or type
                if "house" in self.drum_kit.name.lower():
                    bass_density = 0.4
                    bass_variation = complexity * 0.6
                elif "techno" in self.drum_kit.name.lower():
                    bass_density = 0.3
                    bass_variation = complexity * 0.5
                elif "dubstep" in self.drum_kit.name.lower():
                    bass_density = 0.6
                    bass_variation = complexity * 0.8
                elif "hardstyle" in self.drum_kit.name.lower():
                    bass_density = 0.5
                    bass_variation = complexity * 0.6
                else:
                    bass_density = 0.4
                    bass_variation = complexity * 0.5
                
                # Generate sub-bass pattern
                sub_pattern_length = self.pattern_lengths["sub_bass"]
                sub_pattern = self.generate_pattern("sub_bass", sub_pattern_length, 
                                                  density=bass_density, 
                                                  variation=bass_variation)
                
                # Apply swing to sub-bass pattern if requested
                if swing > 0:
                    sub_pattern = self.apply_swing(sub_pattern, swing)
                
                sub_velocities = self.get_velocity_pattern(sub_pattern, humanize * 0.5)
                
                # Calculate steps per beat for sub-bass
                sub_steps_per_beat = sub_pattern_length // 4
                
                # Render sub-bass pattern for all bars
                for bar in range(bars):
                    bar_position = bar * self.bar_length
                    sub_audio = self.render_pattern(sub_pattern, sub_velocities, 
                                                 sub_bass, sub_steps_per_beat)
                    output[bar_position:bar_position + len(sub_audio)] += sub_audio * 0.8  # Slightly lower volume
        
        # Normalize the output
        if np.max(np.abs(output)) > 0:
            output = output / np.max(np.abs(output)) * 0.95
        
        return output
    
    def save_beat(self, output, filename="edm_beat.wav"):
        """Save the beat as a WAV file"""
        wavfile.write(filename, self.sample_rate, (output * 32767).astype(np.int16))
        print(f"Beat saved to {filename}")
    
    def visualize_beat(self, output):
        """Visualize the waveform and spectrogram of the beat"""
        plt.figure(figsize=(12, 8))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(output)) / self.sample_rate, output)
        plt.title('EDM Beat Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(alpha=0.3)
        
        # Add beat markers
        for i in range(0, len(output), self.beat_length):
            plt.axvline(x=i / self.sample_rate, color='r', linestyle='--', alpha=0.2)
        
        # Plot spectrogram
        plt.subplot(2, 1, 2)
        plt.specgram(output, NFFT=1024, Fs=self.sample_rate, noverlap=512, cmap='inferno')
        plt.title('EDM Beat Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Intensity (dB)')
        plt.ylim(0, 10000)  # Limit frequency range to focus on important parts
        
        plt.tight_layout()
        plt.show()

# Functions to generate beats with different presets

def generate_house_beat(bpm=126, bars=4, complexity=0.5, swing=0.0, humanize=0.2, sub_bass_type=None, waveform=None):
    """Generate a house beat"""
    kit = get_kit_by_name("house")
    
    # Replace the sub-bass with specified type if provided
    if sub_bass_type:
        # Remove existing sub-bass sounds
        sub_bass_sounds = kit.get_sounds_by_category("sub_bass")
        for sound in sub_bass_sounds:
            kit.sounds.pop(sound.name, None)
        
        # Add the new sub-bass sound
        params = {"base_freq": 40}
        if waveform:
            params["waveform"] = waveform
            
        kit.add_sound(create_sub_bass(sub_bass_type, **params))
    
    generator = BeatGenerator(kit, bpm=bpm)
    beat = generator.generate_beat(bars=bars, complexity=complexity, swing=swing, humanize=humanize)
    return beat, generator

def generate_techno_beat(bpm=130, bars=4, complexity=0.4, swing=0.0, humanize=0.15, sub_bass_type=None, waveform=None):
    """Generate a techno beat"""
    kit = get_kit_by_name("techno")
    
    # Replace the sub-bass with specified type if provided
    if sub_bass_type:
        # Remove existing sub-bass sounds
        sub_bass_sounds = kit.get_sounds_by_category("sub_bass")
        for sound in sub_bass_sounds:
            kit.sounds.pop(sound.name, None)
        
        # Add the new sub-bass sound
        params = {"base_freq": 45}
        if waveform:
            params["waveform"] = waveform
            
        kit.add_sound(create_sub_bass(sub_bass_type, **params))
    
    generator = BeatGenerator(kit, bpm=bpm)
    beat = generator.generate_beat(bars=bars, complexity=complexity, swing=swing, humanize=humanize)
    return beat, generator

def generate_dubstep_beat(bpm=140, bars=4, complexity=0.7, swing=0.0, humanize=0.2, sub_bass_type=None, waveform=None):
    """Generate a dubstep beat"""
    kit = get_kit_by_name("dubstep")
    
    # Replace the sub-bass with specified type if provided
    if sub_bass_type:
        # Remove existing sub-bass sounds
        sub_bass_sounds = kit.get_sounds_by_category("sub_bass")
        for sound in sub_bass_sounds:
            kit.sounds.pop(sound.name, None)
        
        # Add the new sub-bass sound
        params = {"base_freq": 40}
        if sub_bass_type == "wobble":
            params["wobble_rate"] = 4.0
            params["format_type"] = "dubstep"
        if waveform:
            params["waveform"] = waveform
        
        kit.add_sound(create_sub_bass(sub_bass_type, **params))
    
    generator = BeatGenerator(kit, bpm=bpm)
    beat = generator.generate_beat(bars=bars, complexity=complexity, swing=swing, humanize=humanize)
    return beat, generator

def generate_hardstyle_beat(bpm=150, bars=4, complexity=0.6, swing=0.0, humanize=0.15, sub_bass_type=None, waveform=None):
    """Generate a hardstyle beat"""
    kit = get_kit_by_name("hardstyle")
    
    # Replace the sub-bass with specified type if provided
    if sub_bass_type:
        # Remove existing sub-bass sounds
        sub_bass_sounds = kit.get_sounds_by_category("sub_bass")
        for sound in sub_bass_sounds:
            kit.sounds.pop(sound.name, None)
        
        # Add the new sub-bass sound
        params = {"base_freq": 50}
        if sub_bass_type == "808":
            params["format_type"] = "modern"
        if waveform:
            params["waveform"] = waveform
        
        kit.add_sound(create_sub_bass(sub_bass_type, **params))
    
    generator = BeatGenerator(kit, bpm=bpm)
    beat = generator.generate_beat(bars=bars, complexity=complexity, swing=swing, humanize=humanize)
    return beat, generator

def generate_custom_beat(bpm=128, kit_name="house", bars=4, complexity=0.5, swing=0.0, humanize=0.2):
    """Generate a beat with the specified kit and parameters"""
    kit = get_kit_by_name(kit_name)
    generator = BeatGenerator(kit, bpm=bpm)
    beat = generator.generate_beat(bars=bars, complexity=complexity, swing=swing, humanize=humanize)
    return beat, generator

# Example usage
if __name__ == "__main__":
    # Generate and save a house beat
    house_beat, house_generator = generate_house_beat(bpm=126, complexity=0.6)
    house_generator.save_beat(house_beat, "house_beat.wav")
    house_generator.visualize_beat(house_beat)
    
    # Generate and save a techno beat
    techno_beat, techno_generator = generate_techno_beat(bpm=130, complexity=0.5)
    techno_generator.save_beat(techno_beat, "techno_beat.wav")
    
    # Generate and save a dubstep beat
    dubstep_beat, dubstep_generator = generate_dubstep_beat(bpm=140, complexity=0.7)
    dubstep_generator.save_beat(dubstep_beat, "dubstep_beat.wav")
