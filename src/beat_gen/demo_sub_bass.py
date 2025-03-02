import numpy as np
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt

from beat_gen.edm_drum_library import (get_kit_by_name, create_sub_bass, DrumKit)
from beat_gen.edm_beat_generator import BeatGenerator
from beat_gen.sub_bass_generator import visualize_sub_bass

def create_output_folder():
    """Create output folder if it doesn't exist"""
    if not os.path.exists("output"):
        os.makedirs("output")
    return "output"

def demo_sub_bass_beats():
    """Generate beats with different sub-bass types"""
    output_folder = create_output_folder()
    
    # Create a base kit
    kit = get_kit_by_name("house")
    
    # Replace the sub-bass with different styles
    bass_types = {
        "sine": create_sub_bass("basic", waveform="sine", base_freq=40),
        "triangle": create_sub_bass("basic", waveform="triangle", base_freq=40),
        "808": create_sub_bass("808", base_freq=40),
        "wobble": create_sub_bass("wobble", wobble_rate=2.0, duration=1.0),
        "reese": create_sub_bass("reese", num_oscillators=4, duration=1.0)
    }
    
    # Generate a beat with each sub-bass type
    for name, sub_bass in bass_types.items():
        print(f"Generating beat with {name} sub-bass...")
        
        # Create a new kit with this sub-bass
        custom_kit = DrumKit(f"Custom {name.capitalize()} Kit")
        
        # Add standard drums
        for sound in kit.get_sounds_by_category("kick"):
            custom_kit.add_sound(sound)
        for sound in kit.get_sounds_by_category("snare"):
            custom_kit.add_sound(sound)
        for sound in kit.get_sounds_by_category("hat"):
            custom_kit.add_sound(sound)
        
        # Add the sub-bass
        custom_kit.add_sound(sub_bass)
        
        # Create beat generator
        generator = BeatGenerator(custom_kit, bpm=126)
        
        # Generate and save beat
        beat = generator.generate_beat(
            bars=8,
            complexity=0.6, 
            swing=0.1, 
            humanize=0.2
        )
        
        # Save the beat
        generator.save_beat(beat, f"{output_folder}/sub_bass_{name}_beat.wav")

def demo_sub_bass_in_different_genres():
    """Demonstrate sub-bass in different genres"""
    output_folder = create_output_folder()
    
    # Generate different genre beats that use the sub-bass
    genres = ["house", "techno", "dubstep", "hardstyle"]
    
    for genre in genres:
        print(f"Generating {genre} beat with sub-bass...")
        
        # Get the kit
        kit = get_kit_by_name(genre)
        
        # Create beat generator
        generator = BeatGenerator(kit, bpm=128)
        
        # Generate and save beat
        beat = generator.generate_beat(
            bars=8,
            complexity=0.65,  # Higher complexity to ensure sub-bass is active
            swing=0.1 if genre in ["house", "techno"] else 0.0,
            humanize=0.2
        )
        
        # Save the beat
        generator.save_beat(beat, f"{output_folder}/{genre}_with_sub_bass.wav")

def visualize_sub_bass_samples():
    """Visualize the waveform and spectrum of sub-bass sounds"""
    # Create sub-bass sounds
    sine_bass = create_sub_bass("basic", waveform="sine", duration=2.0)
    wobble_bass = create_sub_bass("wobble", duration=2.0)
    bass_808 = create_sub_bass("808", duration=2.0)
    reese_bass = create_sub_bass("reese", duration=2.0)
    
    # Visualize
    visualize_sub_bass(sine_bass.audio_data, title="Sine Sub-Bass")
    visualize_sub_bass(wobble_bass.audio_data, title="Wobble Sub-Bass")
    visualize_sub_bass(bass_808.audio_data, title="808 Sub-Bass")
    visualize_sub_bass(reese_bass.audio_data, title="Reese Sub-Bass")

if __name__ == "__main__":
    print("Sub-Bass Integration Demo")
    print("=========================")
    
    # Generate beats with different sub-bass types
    demo_sub_bass_beats()
    
    # Generate different genre beats with sub-bass
    demo_sub_bass_in_different_genres()
    
    # To visualize sub-bass samples (uncomment if visualization is needed)
    # visualize_sub_bass_samples()
    
    print("\nAll demos completed. Output files are in the 'output' folder.")