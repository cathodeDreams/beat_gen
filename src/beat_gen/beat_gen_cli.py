#!/usr/bin/env python
"""
Beat Generator CLI - Generate EDM beats from the command line.

Usage:
    python beat_gen_cli.py --genre=house --bpm=126 --bars=4 --complexity=0.6

This script provides command-line access to the beat-gen package's drum pattern
generation capabilities, allowing you to create EDM beats in various styles
without using the web interface.
"""

import os
import argparse
import numpy as np
from scipy.io import wavfile

# Import beat-gen components
from beat_gen.edm_drum_library import get_kit_by_name
from beat_gen.edm_beat_generator import (
    BeatGenerator,
    generate_house_beat,
    generate_techno_beat,
    generate_dubstep_beat,
    generate_hardstyle_beat
)

def parse_time_signature(time_sig_str):
    """Parse time signature string (e.g. '4/4') to a tuple (e.g. (4, 4))"""
    if not time_sig_str or '/' not in time_sig_str:
        return (4, 4)  # Default to 4/4
    
    numerator, denominator = time_sig_str.split('/')
    return (int(numerator), int(denominator))

def generate_beat(args):
    """Generate a beat based on the provided arguments"""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse time signature
    time_signature = parse_time_signature(args.time_signature)
    
    # Prepare kwargs for beat generation
    kwargs = {
        'bpm': args.bpm,
        'bars': args.bars,
        'complexity': args.complexity,
        'swing': args.swing,
        'humanize': args.humanize,
        'time_signature': time_signature,
        'use_euclidean': args.use_euclidean,
        'evolving': args.evolving
    }
    
    # Add sub-bass type if specified
    if args.sub_bass_type:
        kwargs['sub_bass_type'] = args.sub_bass_type
    
    # Add waveform if specified (for basic sub-bass)
    if args.waveform:
        kwargs['waveform'] = args.waveform
    
    # Generate beat based on genre
    print(f"Generating {args.genre} beat...")
    
    if args.genre == 'house':
        beat, generator = generate_house_beat(**kwargs)
    elif args.genre == 'techno':
        beat, generator = generate_techno_beat(**kwargs)
    elif args.genre == 'dubstep':
        beat, generator = generate_dubstep_beat(**kwargs)
    elif args.genre == 'hardstyle':
        beat, generator = generate_hardstyle_beat(**kwargs)
    else:
        # Fallback to generic beat generation with specified kit
        kit = get_kit_by_name(args.genre)
        generator = BeatGenerator(kit, bpm=args.bpm, time_signature=time_signature)
        beat = generator.generate_beat(
            bars=args.bars, 
            complexity=args.complexity,
            swing=args.swing, 
            humanize=args.humanize
        )
    
    # Apply advanced fill if requested
    if args.add_fill:
        print(f"Adding {args.fill_type} fill...")
        # Calculate the position for the fill (last beat of the last bar)
        fill_position = generator.bar_length * (args.bars - 1) + generator.beat_length * 3
        
        # Make sure the fill position is valid
        if fill_position < len(beat):
            # Generate the fill
            fill_audio = generator.generate_advanced_fill(1, args.fill_type, args.fill_intensity)
            
            # Ensure we don't go out of bounds
            end_pos = min(fill_position + len(fill_audio), len(beat))
            beat[fill_position:end_pos] = fill_audio[:end_pos - fill_position]
    
    # Apply polyrhythm if requested
    if args.add_polyrhythm:
        print(f"Adding {args.rhythm1} against {args.rhythm2} polyrhythm...")
        # Get a percussion sound
        perc_sounds = generator.drum_kit.get_sounds_by_category("percussion")
        
        if perc_sounds:
            # Determine pattern length based on time signature
            beats_per_bar, _ = time_signature
            pattern_length = 16 * beats_per_bar * args.bars // 4  # 16th notes per bar
            
            # Generate polyrhythm pattern
            poly_pattern = generator.generate_polyrhythm(pattern_length, args.rhythm1, args.rhythm2)
            
            # Calculate steps per beat - always 4 for 16th notes
            steps_per_beat = 4  # 4 16th notes per beat
            
            # Create velocities
            velocities = []
            for p in poly_pattern:
                if p == 0:
                    velocities.append(0)
                elif p == 1:
                    velocities.append(0.7)  # Regular hits
                elif p == 2:
                    velocities.append(1.0)  # Accented hits where rhythms coincide
            
            # Render the polyrhythm with the first percussion sound
            perc_sound = perc_sounds[0]
            poly_audio = generator.render_pattern(poly_pattern, velocities, perc_sound, steps_per_beat)
            
            # Ensure the polyrhythm audio has the same length as the main beat
            if len(poly_audio) < len(beat):
                # Pad with zeros if shorter
                padded_poly_audio = np.zeros(len(beat))
                padded_poly_audio[:len(poly_audio)] = poly_audio
                poly_audio = padded_poly_audio
            elif len(poly_audio) > len(beat):
                # Truncate if longer
                poly_audio = poly_audio[:len(beat)]
            
            # Mix it into the beat at 70% volume
            beat += poly_audio * 0.7
    
    # Normalize the final beat
    if np.max(np.abs(beat)) > 0:
        beat = beat / np.max(np.abs(beat)) * 0.95
    
    # Generate the output filename
    filename = f"{args.genre}_{args.bpm}bpm_{args.bars}bars"
    if args.complexity != 0.5:
        filename += f"_complex{int(args.complexity*10)}"
    if args.swing > 0:
        filename += f"_swing{int(args.swing*10)}"
    if args.output_name:
        # Use custom name if provided
        output_path = os.path.join(args.output_dir, f"{args.output_name}.wav")
    else:
        # Use auto-generated name
        output_path = os.path.join(args.output_dir, f"{filename}.wav")
    
    # Save the beat
    generator.save_beat(beat, output_path)
    
    # Generate visualization if requested
    if args.visualize:
        print("Generating visualization...")
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(beat)) / generator.sample_rate, beat)
        plt.title(f'{args.genre.title()} Beat Waveform ({args.bpm} BPM)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(alpha=0.3)
        
        # Add beat markers
        for i in range(0, len(beat), generator.beat_length):
            plt.axvline(x=i / generator.sample_rate, color='r', linestyle='--', alpha=0.2)
        
        # Plot spectrogram
        plt.subplot(2, 1, 2)
        plt.specgram(beat, NFFT=1024, Fs=generator.sample_rate, noverlap=512, cmap='inferno')
        plt.title(f'{args.genre.title()} Beat Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Intensity (dB)')
        plt.ylim(0, 10000)  # Limit frequency range to focus on important parts
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.splitext(output_path)[0] + '_viz.png'
        plt.savefig(viz_path)
        plt.close()
        print(f"Visualization saved to: {viz_path}")
    
    print(f"Beat saved to: {output_path}")
    
    # Play the beat if requested
    if args.play:
        try:
            print("Playing beat...")
            import simpleaudio as sa
            audio_data = (beat * 32767).astype(np.int16)
            play_obj = sa.play_buffer(audio_data, 1, 2, generator.sample_rate)
            play_obj.wait_done()
        except ImportError:
            print("Cannot play audio: simpleaudio package not installed.")
            print("Install with: pip install simpleaudio")

def main():
    """Main function to parse arguments and generate beat"""
    parser = argparse.ArgumentParser(
        description="Generate EDM beats from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic beat parameters
    parser.add_argument("--genre", type=str, default="house", 
                        choices=["house", "techno", "dubstep", "hardstyle"],
                        help="Genre of beat to generate")
    parser.add_argument("--bpm", type=int, default=128,
                        help="Beats per minute")
    parser.add_argument("--bars", type=int, default=4,
                        help="Number of bars to generate")
    parser.add_argument("--time-signature", type=str, default="4/4",
                        help="Time signature in the format 'num/denom' (e.g. 4/4, 3/4, 7/8)")
    
    # Groove parameters
    parser.add_argument("--complexity", type=float, default=0.5,
                        help="Complexity of the beat (0.0-1.0)")
    parser.add_argument("--swing", type=float, default=0.0,
                        help="Amount of swing (0.0-1.0)")
    parser.add_argument("--humanize", type=float, default=0.2,
                        help="Amount of humanization (0.0-1.0)")
    parser.add_argument("--use-euclidean", action="store_true",
                        help="Use euclidean rhythms for more interesting patterns")
    parser.add_argument("--evolving", action="store_true",
                        help="Create patterns that evolve over time")
    
    # Sound options
    parser.add_argument("--sub-bass-type", type=str, 
                        choices=["basic", "808", "wobble", "reese"],
                        help="Type of sub-bass to use")
    parser.add_argument("--waveform", type=str,
                        choices=["sine", "triangle", "square", "saw"],
                        help="Waveform type for basic sub-bass")
    
    # Fill options
    parser.add_argument("--add-fill", action="store_true",
                        help="Add an advanced fill at the end of the beat")
    parser.add_argument("--fill-type", type=str, default="buildup",
                        choices=["buildup", "glitch", "roll"],
                        help="Type of fill to add")
    parser.add_argument("--fill-intensity", type=float, default=0.7,
                        help="Intensity of the fill (0.0-1.0)")
    
    # Polyrhythm options
    parser.add_argument("--add-polyrhythm", action="store_true",
                        help="Add a polyrhythm to the beat")
    parser.add_argument("--rhythm1", type=int, default=4,
                        help="First rhythm division for polyrhythm")
    parser.add_argument("--rhythm2", type=int, default=3,
                        help="Second rhythm division for polyrhythm")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save output files")
    parser.add_argument("--output-name", type=str,
                        help="Custom filename for the output (without extension)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate waveform and spectrogram visualization")
    parser.add_argument("--play", action="store_true",
                        help="Play the beat after generating (requires simpleaudio package)")
    
    args = parser.parse_args()
    generate_beat(args)

if __name__ == "__main__":
    main()
