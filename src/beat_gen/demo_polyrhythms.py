import numpy as np
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt

from beat_gen.edm_drum_library import get_kit_by_name
from beat_gen.edm_beat_generator import BeatGenerator

def visualize_polyrhythm(pattern, title):
    """Visualize a polyrhythm pattern"""
    plt.figure(figsize=(12, 2))
    for i, val in enumerate(pattern):
        if val == 1:
            plt.axvline(x=i, color='b', linewidth=2)
        elif val == 2:
            plt.axvline(x=i, color='r', linewidth=3)  # Accent where rhythms coincide
    
    plt.title(title)
    plt.xlabel('Step')
    plt.xlim(0, len(pattern))
    plt.yticks([])
    return plt

def demo_polyrhythms():
    """Demonstrate polyrhythms"""
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Get a kit
    kit = get_kit_by_name("techno")
    
    # Create a beat generator
    generator = BeatGenerator(kit, bpm=130)
    
    # Generate a range of polyrhythms
    rhythm_pairs = [
        (3, 4),   # 3 against 4
        (2, 3),   # 2 against 3
        (4, 5),   # 4 against 5
        (5, 7),   # 5 against 7
        (3, 8)    # 3 against 8
    ]
    
    # Create visualization of each polyrhythm
    fig = plt.figure(figsize=(12, 10))
    for i, (r1, r2) in enumerate(rhythm_pairs):
        pattern_length = 16 * 3  # Make it long enough to see the pattern
        poly_pattern = generator.generate_polyrhythm(pattern_length, r1, r2)
        
        plt.subplot(len(rhythm_pairs), 1, i+1)
        for j, val in enumerate(poly_pattern):
            if val == 1:
                plt.axvline(x=j, color='b', linewidth=2, alpha=0.7)
            elif val == 2:
                plt.axvline(x=j, color='r', linewidth=3)  # Accent where rhythms coincide
        
        plt.title(f'{r1} against {r2} Polyrhythm')
        plt.xlabel('Step')
        plt.xlim(0, len(poly_pattern))
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig("output/polyrhythm_patterns.png")
    
    # Create audio demos for each polyrhythm
    for r1, r2 in rhythm_pairs:
        print(f"Generating {r1} against {r2} polyrhythm...")
        
        # Create the polyrhythm pattern
        pattern_length = 16 * 2  # Two bars of 16th notes
        poly_pattern = generator.generate_polyrhythm(pattern_length, r1, r2)
        
        # Get sounds
        kick = generator.drum_kit.get_sounds_by_category("kick")[0]
        snare = generator.drum_kit.get_sounds_by_category("snare")[0]
        
        # Render the pattern
        steps_per_beat = pattern_length // 8  # 2 bars, 4 beats per bar
        
        # Convert the pattern to velocities
        velocities = []
        for p in poly_pattern:
            if p == 0:
                velocities.append(0)
            elif p == 1:
                velocities.append(0.7)  # Regular hits
            elif p == 2:
                velocities.append(1.0)  # Accented hits where rhythms coincide
        
        # Render with kick for first rhythm and snare for second rhythm
        kick_audio = generator.render_pattern(poly_pattern, velocities, kick, steps_per_beat)
        
        # Render several repetitions of the pattern
        repeats = 4
        full_audio = np.zeros(len(kick_audio) * repeats)
        for i in range(repeats):
            full_audio[i * len(kick_audio):(i + 1) * len(kick_audio)] = kick_audio
        
        # Save the audio
        wavfile.write(
            f"output/polyrhythm_{r1}_against_{r2}.wav", 
            generator.sample_rate, 
            (full_audio * 32767).astype(np.int16)
        )
    
    print("Polyrhythm demos saved to the output directory.")

if __name__ == "__main__":
    demo_polyrhythms()