import numpy as np
import os
from scipy.io import wavfile

from beat_gen.edm_drum_library import get_kit_by_name
from beat_gen.edm_beat_generator import (
    BeatGenerator,
    generate_house_beat,
    generate_techno_beat
)
from beat_gen.city_pop_bassline import CityPopBasslineGenerator

def generate_city_pop_track(
    key="F",
    chord_progression_type="city_pop_2",
    pattern_type="smooth_1",
    bars=8,
    bpm=96,
    complexity=0.7,
    percussion_style="house",
    drum_complexity=0.4,
    bass_roundness=0.7,
    bass_brightness=0.5,
    bass_harmonics=0.6,
    walking_bass=False,
    output_dir="output"
):
    """
    Generate a city-pop inspired track with sophisticated jazz-influenced bassline
    
    Parameters:
    -----------
    key : str
        The musical key for the track (e.g., "C", "F#")
    chord_progression_type : str
        Type of chord progression (e.g., "city_pop_1", "modal_1", "deceptive_1")
    pattern_type : str
        Bassline rhythm pattern to use
    bars : int
        Number of bars to generate
    bpm : int
        Tempo in beats per minute
    complexity : float
        Complexity of the bassline (0.0-1.0)
    percussion_style : str
        Style of drum kit to use ("house", "techno")
    drum_complexity : float
        Complexity of the drum pattern (0.0-1.0)
    bass_roundness : float
        Tone parameter - blend of sine & triangle (0=pure sine, 1=pure triangle)
    bass_brightness : float
        Tone parameter - adds upper harmonics
    bass_harmonics : float
        Tone parameter - amount of additional harmonic content
    walking_bass : bool
        Whether to use walking bass style
    output_dir : str
        Directory to save output files
    
    Returns:
    --------
    tuple
        (mixed_audio, beat_audio, bassline_audio)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate drums with light swing for that city-pop feel
    print(f"Generating {percussion_style} beat at {bpm} BPM...")
    
    # Use house or techno kit for light percussion
    if percussion_style == "techno":
        beat, beat_generator = generate_techno_beat(
            bpm=bpm,
            bars=bars,
            complexity=drum_complexity,
            swing=0.15,  # Light swing for city pop feel
            humanize=0.25
        )
    else:  # Default to house
        beat, beat_generator = generate_house_beat(
            bpm=bpm,
            bars=bars,
            complexity=drum_complexity,
            swing=0.15,  # Light swing for city pop feel
            humanize=0.25
        )
    
    # 2. Generate the city-pop bassline
    print(f"Generating city-pop bassline in {key}...")
    
    # Configure bass tone parameters
    bass_params = {
        "roundness": bass_roundness,
        "brightness": bass_brightness,
        "harmonics": bass_harmonics
    }
    
    # Create city pop bassline generator
    city_pop_generator = CityPopBasslineGenerator(sample_rate=beat_generator.sample_rate)
    
    # Generate the bassline
    bassline, sequence, chord_progression = city_pop_generator.generate_city_pop_bassline(
        key=key,
        chord_progression_type=chord_progression_type,
        pattern_type=pattern_type,
        bars=bars,
        complexity=complexity,
        walking_bass=walking_bass,
        bass_params=bass_params
    )
    
    # Adjust bassline length to match beat length
    if len(bassline) > len(beat):
        bassline = bassline[:len(beat)]
    elif len(bassline) < len(beat):
        padding = np.zeros(len(beat) - len(bassline))
        bassline = np.concatenate([bassline, padding])
    
    # 3. Mix beat and bassline
    mixed = beat * 0.65 + bassline * 0.85  # Emphasize the bass a bit more
    
    # Normalize the final mix
    mixed = mixed / np.max(np.abs(mixed)) * 0.95
    
    # 4. Save individual components and the mixed track
    beat_filename = os.path.join(output_dir, f"city_pop_beat_{bpm}bpm.wav")
    bassline_filename = os.path.join(output_dir, f"city_pop_bassline_{key}_{bpm}bpm.wav")
    mixed_filename = os.path.join(output_dir, f"city_pop_track_{key}_{bpm}bpm.wav")
    
    # Print chord progression for reference
    print("\nChord Progression:")
    for i, chord in enumerate(chord_progression, 1):
        print(f"Bar {i}: {', '.join(chord)}")
    
    # Save files
    beat_generator.save_beat(beat, beat_filename)
    city_pop_generator.save_bassline(bassline, bassline_filename)
    wavfile.write(mixed_filename, beat_generator.sample_rate, (mixed * 32767).astype(np.int16))
    
    print(f"\nTrack saved to: {mixed_filename}")
    
    return mixed, beat, bassline

# Add to beat_gen_cli.py
def add_city_pop_to_cli_arguments(parser):
    """Add city-pop generation arguments to an existing ArgumentParser"""
    # Add a city-pop command group
    city_pop_group = parser.add_argument_group('City Pop Options')
    
    city_pop_group.add_argument("--city-pop", action="store_true",
                        help="Generate a city-pop style track with jazz-influenced bassline")
    
    city_pop_group.add_argument("--key", type=str, default="F",
                        help="Musical key for the track (e.g., C, F#)")
    
    city_pop_group.add_argument("--chord-progression", type=str, default="city_pop_2",
                        choices=["city_pop_1", "city_pop_2", "city_pop_3", 
                                "modal_1", "modal_2", "deceptive_1", 
                                "chromatic_1", "minor_jazz"],
                        help="Type of chord progression to use")
    
    city_pop_group.add_argument("--bass-pattern", type=str, default="smooth_1",
                        choices=["smooth_1", "smooth_2", "jazzy_1", 
                                "fusion_1", "walking", "chromatic"],
                        help="Bassline rhythm pattern to use")
    
    city_pop_group.add_argument("--walking-bass", action="store_true",
                        help="Use walking bass style instead of pattern-based")
    
    city_pop_group.add_argument("--bass-complexity", type=float, default=0.7,
                        help="Complexity of the bassline (0.0-1.0)")
    
    city_pop_group.add_argument("--bass-roundness", type=float, default=0.7,
                        help="Tone: blend of sine/triangle (0=sine, 1=triangle)")
    
    city_pop_group.add_argument("--bass-brightness", type=float, default=0.5,
                        help="Tone: adds upper harmonics (0.0-1.0)")
    
    city_pop_group.add_argument("--bass-harmonics", type=float, default=0.6,
                        help="Tone: amount of additional harmonic content (0.0-1.0)")
    
    return parser

# Example of how to extend the main function in beat_gen_cli.py
def extended_main():
    """Extended main function with city-pop support"""
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Generate EDM beats or city-pop tracks from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic beat parameters
    parser.add_argument("--genre", type=str, default="house", 
                    choices=["house", "techno", "dubstep", "hardstyle"],
                    help="Genre of beat to generate")
    # ... (existing beat generator arguments)
    
    # Add city-pop options
    parser = add_city_pop_to_cli_arguments(parser)
    
    args = parser.parse_args()
    
    # Check if city-pop mode is requested
    if args.city_pop:
        generate_city_pop_track(
            key=args.key,
            chord_progression_type=args.chord_progression,
            pattern_type=args.bass_pattern,
            bars=args.bars,
            bpm=args.bpm,
            complexity=args.bass_complexity,
            percussion_style=args.genre,
            drum_complexity=args.complexity,
            bass_roundness=args.bass_roundness,
            bass_brightness=args.bass_brightness,
            bass_harmonics=args.bass_harmonics,
            walking_bass=args.walking_bass,
            output_dir=args.output_dir
        )
    else:
        # Original beat generation logic
        generate_beat(args)

# Example usage
if __name__ == "__main__":
    import argparse  # Added for the example
    
    # If run directly, generate a demo city-pop track
    generate_city_pop_track(
        key="F",
        chord_progression_type="city_pop_2",
        pattern_type="smooth_2",
        bars=8,
        bpm=96,
        complexity=0.7,
        bass_roundness=0.7,
        bass_brightness=0.5,
        bass_harmonics=0.6
    )
