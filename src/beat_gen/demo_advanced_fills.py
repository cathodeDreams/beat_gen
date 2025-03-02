import numpy as np
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt

from beat_gen.edm_drum_library import get_kit_by_name
from beat_gen.edm_beat_generator import BeatGenerator

def demo_advanced_fills():
    """Demonstrate the different types of advanced fills"""
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Get a kit
    kit = get_kit_by_name("dubstep")  # Dubstep has more dramatic percussion
    
    # Create a beat generator
    generator = BeatGenerator(kit, bpm=140)
    
    # Generate and save different fill types
    fill_types = ["buildup", "glitch", "roll"]
    
    # Generate each fill type with 2 beats length
    for fill_type in fill_types:
        print(f"Generating {fill_type} fill...")
        
        # Generate a fill with 2 beat length
        fill_audio = generator.generate_advanced_fill(
            length=2,
            fill_type=fill_type,
            intensity=0.8
        )
        
        # Save the fill
        wavfile.write(
            f"output/{fill_type}_fill.wav", 
            generator.sample_rate, 
            (fill_audio * 32767).astype(np.int16)
        )
        
        # Plot the fill waveform
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(fill_audio)) / generator.sample_rate, fill_audio)
        plt.title(f'{fill_type.title()} Fill Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(alpha=0.3)
        plt.savefig(f"output/{fill_type}_fill_waveform.png")
        plt.close()
    
    # Generate a beat with each fill type
    for fill_type in fill_types:
        print(f"Generating beat with {fill_type} fills...")
        
        # Create a 4-bar beat with normal generation
        beat = np.zeros(generator.bar_length * 4)
        
        # Add regular drum pattern for first 3 bars
        for bar in range(3):
            bar_beat = generator.generate_beat(bars=1, complexity=0.6)
            beat[bar * generator.bar_length:(bar + 1) * generator.bar_length] = bar_beat
        
        # Add a bar with the fill at the end
        last_bar_beat = generator.generate_beat(bars=1, complexity=0.6)
        
        # Only use the first 3/4 of the last bar
        last_bar_segment = last_bar_beat[:3 * generator.beat_length]
        position = 3 * generator.bar_length
        beat[position:position + len(last_bar_segment)] += last_bar_segment
        
        # Add the fill for the last beat
        fill_position = 3 * generator.bar_length + 3 * generator.beat_length
        fill_audio = generator.generate_advanced_fill(1, fill_type, 0.8)
        beat[fill_position:fill_position + len(fill_audio)] += fill_audio
        
        # Normalize
        if np.max(np.abs(beat)) > 0:
            beat = beat / np.max(np.abs(beat)) * 0.95
        
        # Save the beat
        wavfile.write(
            f"output/beat_with_{fill_type}_fill.wav", 
            generator.sample_rate, 
            (beat * 32767).astype(np.int16)
        )
    
    print("Advanced fill demos saved to the output directory.")

if __name__ == "__main__":
    demo_advanced_fills()