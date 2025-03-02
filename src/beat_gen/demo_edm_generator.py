import numpy as np
from scipy.io import wavfile
import os

from beat_gen.edm_drum_library import (get_kit_by_name, create_kick,
                            create_snare, create_hat, create_percussion, DrumKit)
from beat_gen.edm_beat_generator import (BeatGenerator, generate_house_beat, generate_techno_beat,
                              generate_dubstep_beat, generate_hardstyle_beat)

def create_output_folder():
    """Create output folder if it doesn't exist"""
    if not os.path.exists("output"):
        os.makedirs("output")
    return "output"

def demo_preset_beats():
    """Generate all preset beats"""
    output_folder = create_output_folder()
    
    # Generate house beat
    print("Generating house beat...")
    house_beat, house_generator = generate_house_beat(bpm=126, bars=8, complexity=0.6, swing=0.1)
    house_generator.save_beat(house_beat, f"{output_folder}/house_beat.wav")
    
    # Generate techno beat
    print("Generating techno beat...")
    techno_beat, techno_generator = generate_techno_beat(bpm=130, bars=8, complexity=0.5)
    techno_generator.save_beat(techno_beat, f"{output_folder}/techno_beat.wav")
    
    # Generate dubstep beat
    print("Generating dubstep beat...")
    dubstep_beat, dubstep_generator = generate_dubstep_beat(bpm=140, bars=8, complexity=0.7)
    dubstep_generator.save_beat(dubstep_beat, f"{output_folder}/dubstep_beat.wav")
    
    # Generate hardstyle beat
    print("Generating hardstyle beat...")
    hardstyle_beat, hardstyle_generator = generate_hardstyle_beat(bpm=150, bars=8, complexity=0.6)
    hardstyle_generator.save_beat(hardstyle_beat, f"{output_folder}/hardstyle_beat.wav")

def demo_complexity_variations():
    """Generate beats with different complexity levels"""
    output_folder = create_output_folder()
    kit = get_kit_by_name("house")
    
    for complexity in [0.2, 0.5, 0.8]:
        print(f"Generating beat with complexity {complexity}...")
        generator = BeatGenerator(kit, bpm=126)
        beat = generator.generate_beat(bars=8, complexity=complexity, swing=0.1)
        generator.save_beat(beat, f"{output_folder}/house_complexity_{int(complexity*10)}.wav")

def demo_swing_variations():
    """Generate beats with different swing amounts"""
    output_folder = create_output_folder()
    kit = get_kit_by_name("house")
    
    for swing in [0.0, 0.3, 0.6]:
        print(f"Generating beat with swing {swing}...")
        generator = BeatGenerator(kit, bpm=126)
        beat = generator.generate_beat(bars=8, complexity=0.5, swing=swing)
        generator.save_beat(beat, f"{output_folder}/house_swing_{int(swing*10)}.wav")

def demo_custom_kit():
    """Create and use a custom drum kit"""
    output_folder = create_output_folder()
    
    # Create a custom kit
    print("Creating custom kit...")
    kit = DrumKit("Custom Kit")
    
    # Add a custom kick
    kit.add_sound(create_kick("deep_house", freq_start=130, freq_end=30, distortion=0.4))
    
    # Add a layered snare
    kit.add_sound(create_snare("layered"))
    
    # Add hats
    kit.add_sound(create_hat("closed", freq_range=(3000, 15000), decay=40))
    kit.add_sound(create_hat("open", freq_range=(2000, 12000), decay=15))
    
    # Add percussion
    kit.add_sound(create_percussion("click", duration=0.12))
    kit.add_sound(create_percussion("blip", duration=0.15))
    
    # Generate beat with custom kit
    print("Generating beat with custom kit...")
    generator = BeatGenerator(kit, bpm=125)
    beat = generator.generate_beat(bars=8, complexity=0.6, swing=0.2)
    generator.save_beat(beat, f"{output_folder}/custom_kit_beat.wav")

def demo_visualization():
    """Generate a beat and display its waveform and spectrogram"""
    print("Generating beat for visualization...")
    dubstep_beat, dubstep_generator = generate_dubstep_beat(bpm=140, bars=8, complexity=0.7)
    dubstep_generator.visualize_beat(dubstep_beat)

if __name__ == "__main__":
    print("EDM Beat Generator Demo")
    print("======================")
    
    # Demo different preset beats
    demo_preset_beats()
    
    # Demo complexity variations
    demo_complexity_variations()
    
    # Demo swing variations
    demo_swing_variations()
    
    # Demo custom kit
    demo_custom_kit()
    
    # Demo visualization (will open a matplotlib window)
    demo_visualization()
    
    print("\nAll demos completed. Output files are in the 'output' folder.")
