from beat_gen.edm_drum_library import get_kit_by_name
from beat_gen.edm_beat_generator import (
    BeatGenerator, 
    generate_house_beat, 
    generate_techno_beat,
    generate_dubstep_beat, 
    generate_hardstyle_beat
)
import os

def generate_basic_beat():
    """Generate a basic beat using the standard approach"""
    # Get a preset kit
    kit = get_kit_by_name("techno")

    # Create a beat generator
    generator = BeatGenerator(kit, bpm=128)

    # Generate a beat
    beat = generator.generate_beat(
        bars=8,           # Generate 8 bars
        complexity=0.6,   # Medium-high complexity
        swing=0.1,        # Slight swing feel
        humanize=0.2      # Natural timing variations
    )

    # Save the beat
    generator.save_beat(beat, "output/basic_beat.wav")

    # Visualize the beat
    generator.visualize_beat(beat)

def demo_new_features():
    """
    Demonstrate the new rhythmic features added to the beat generator
    """
    print("Generating beats with new rhythmic features...")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # 1. Euclidean rhythms in house music
    print("Generating house beat with euclidean rhythms...")
    euclidean_house, euclidean_house_generator = generate_house_beat(
        bpm=126, complexity=0.6, use_euclidean=True
    )
    euclidean_house_generator.save_beat(euclidean_house, "output/euclidean_house_beat.wav")
    
    # 2. Different time signature (7/8) for techno
    print("Generating techno beat with 7/8 time signature...")
    techno_78, techno_78_generator = generate_techno_beat(
        bpm=130, complexity=0.5, time_signature=(7, 8), use_euclidean=True
    )
    techno_78_generator.save_beat(techno_78, "output/techno_7_8_beat.wav")
    
    # 3. House with evolving patterns
    print("Generating house beat with evolving patterns...")
    evolving_house, evolving_house_generator = generate_house_beat(
        bpm=124, complexity=0.7, evolving=True
    )
    evolving_house_generator.save_beat(evolving_house, "output/evolving_house_beat.wav")
    
    # 4. Techno with polyrhythms
    print("Generating complex techno beat with high complexity...")
    polyrhythm_techno, polyrhythm_techno_generator = generate_techno_beat(
        bpm=135, complexity=0.8, use_euclidean=True
    )
    polyrhythm_techno_generator.save_beat(polyrhythm_techno, "output/polyrhythm_techno_beat.wav")
    
    print("Demo beats saved to the output directory.")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Run the demo of new features
    demo_new_features()
    
    # Generate a basic beat
    # generate_basic_beat()
