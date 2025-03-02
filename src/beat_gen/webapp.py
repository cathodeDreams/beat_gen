import os
from flask import Flask, render_template, request, send_file, url_for, jsonify
import numpy as np
from scipy.io import wavfile
import tempfile
import uuid
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from beat_gen.edm_drum_library import get_kit_by_name
from beat_gen.edm_beat_generator import (
    BeatGenerator, 
    generate_house_beat, 
    generate_techno_beat, 
    generate_dubstep_beat, 
    generate_hardstyle_beat
)

app = Flask(__name__)

# Create templates folder if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), "templates"), exist_ok=True)

# Create static folder if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)

# Create a temporary directory for storing generated beats
TEMP_DIR = os.path.join(tempfile.gettempdir(), "beat_gen_webapp")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def parse_time_signature(time_sig_str):
    """Parse time signature string (e.g. '4/4') to a tuple (e.g. (4, 4))"""
    if not time_sig_str or '/' not in time_sig_str:
        return (4, 4)  # Default to 4/4
    
    numerator, denominator = time_sig_str.split('/')
    return (int(numerator), int(denominator))

def generate_beat_from_params(request_data, preview=False):
    """
    Common function to generate a beat from request parameters.
    Returns the beat audio data and generator.
    """
    # Get form parameters
    genre = request_data.get('genre', 'house')
    bpm = int(request_data.get('bpm', 128))
    bars = int(request_data.get('bars', 4))
    complexity = float(request_data.get('complexity', 0.5))
    swing = float(request_data.get('swing', 0.0))
    humanize = float(request_data.get('humanize', 0.2))
    
    # Advanced parameters
    time_signature = parse_time_signature(request_data.get('time_signature', '4/4'))
    use_euclidean = request_data.get('use_euclidean') == 'true'
    evolving = request_data.get('evolving') == 'true'
    
    # Sound options
    sub_bass_type = request_data.get('sub_bass_type', 'basic')
    waveform = request_data.get('waveform', None)
    wobble_rate = float(request_data.get('wobble_rate', 2.0))
    
    # Fill options
    use_advanced_fills = request_data.get('use_advanced_fills') == 'true'
    fill_type = request_data.get('fill_type', 'buildup')
    fill_intensity = float(request_data.get('fill_intensity', 0.7))
    
    # Polyrhythm options
    use_polyrhythm = request_data.get('use_polyrhythm') == 'true'
    rhythm1 = int(request_data.get('rhythm1', 4))
    rhythm2 = int(request_data.get('rhythm2', 3))
    
    # If preview is requested, generate a shorter beat
    if preview:
        bars = min(bars, 2)  # Limit to 2 bars for preview
    
    # Generate beat based on genre with all advanced parameters
    beat = None
    generator = None
    
    # Prepare kwargs for beat generation
    kwargs = {
        'bpm': bpm,
        'bars': bars,
        'complexity': complexity,
        'swing': swing,
        'humanize': humanize,
        'sub_bass_type': sub_bass_type,
        'time_signature': time_signature,
        'use_euclidean': use_euclidean,
        'evolving': evolving
    }
    
    # Add waveform only if it's specified
    if waveform:
        kwargs['waveform'] = waveform
    
    # Special bass parameters should be passed directly to the sub_bass creation function
    # instead of to the beat generation function, so we should NOT include wobble_rate here
    
    if genre == 'house':
        beat, generator = generate_house_beat(**kwargs)
    elif genre == 'techno':
        beat, generator = generate_techno_beat(**kwargs)
    elif genre == 'dubstep':
        beat, generator = generate_dubstep_beat(**kwargs)
    elif genre == 'hardstyle':
        beat, generator = generate_hardstyle_beat(**kwargs)
    else:
        # Use custom kit
        kit = get_kit_by_name(genre)
        generator = BeatGenerator(kit, bpm=bpm, time_signature=time_signature)
        beat = generator.generate_beat(
            bars=bars, complexity=complexity, 
            swing=swing, humanize=humanize,
            time_signature=time_signature
        )
    
    # Apply advanced fills if requested
    if use_advanced_fills and bars > 0:  # Make sure we have at least one bar
        # Calculate the position for the fill (last beat of the last bar)
        fill_position = generator.bar_length * (bars - 1) + generator.beat_length * 3
        
        # Make sure the fill position is valid
        if fill_position < len(beat):
            # Generate the fill
            fill_audio = generator.generate_advanced_fill(1, fill_type, fill_intensity)
            
            # Ensure we don't go out of bounds
            end_pos = min(fill_position + len(fill_audio), len(beat))
            beat[fill_position:end_pos] = fill_audio[:end_pos - fill_position]
    
    # Apply polyrhythm if requested
    if use_polyrhythm:
        # Create a polyrhythm pattern for percussion
        # First, get a percussion sound
        perc_sounds = generator.drum_kit.get_sounds_by_category("percussion")
        
        if perc_sounds:
            # Determine pattern length based on time signature
            beats_per_bar, _ = time_signature
            pattern_length = 16 * beats_per_bar * bars // 4  # 16th notes per bar, adjusted for time signature
            
            # Create the polyrhythm pattern
            poly_pattern = generator.generate_polyrhythm(pattern_length, rhythm1, rhythm2)
            
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
            
            # Make sure we have a percussion sound
            perc_sound = perc_sounds[0]
            
            try:
                # Render the polyrhythm with the percussion sound
                poly_audio = generator.render_pattern(poly_pattern, velocities, perc_sound, steps_per_beat)
            except Exception as e:
                # If there's an error, log it and continue without polyrhythm
                print(f"Error generating polyrhythm: {e}")
                poly_audio = None
            
            # Only mix if we successfully generated poly_audio
            if poly_audio is not None:
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
    
    # Normalize audio
    if np.max(np.abs(beat)) > 0:
        beat = beat / np.max(np.abs(beat)) * 0.95
    
    return beat, generator

@app.route('/generate', methods=['POST'])
def generate():
    """Generate and download a beat based on form parameters"""
    # Generate the beat
    beat, generator = generate_beat_from_params(request.form)
    
    # Include visualization if requested
    generate_visualization = request.form.get('generate_visualization') == 'true'
    
    if generate_visualization:
        # Create visualization
        visualize_beat_to_file(beat, generator)
    
    # Generate unique filename
    filename = f"{request.form.get('genre', 'beat')}_{request.form.get('bpm', '128')}bpm_{uuid.uuid4().hex[:8]}.wav"
    filepath = os.path.join(TEMP_DIR, filename)
    
    # Save beat
    wavfile.write(filepath, generator.sample_rate, (beat * 32767).astype(np.int16))
    
    return send_file(
        filepath,
        as_attachment=True,
        download_name=filename,
        mimetype='audio/wav'
    )

@app.route('/preview', methods=['POST'])
def preview():
    """Generate a preview beat (shorter) based on form parameters"""
    # Generate the beat with preview flag
    beat, generator = generate_beat_from_params(request.form, preview=True)
    
    # Generate unique filename for the preview
    preview_filename = f"preview_{uuid.uuid4().hex[:8]}.wav"
    preview_filepath = os.path.join(TEMP_DIR, preview_filename)
    
    # Save preview beat
    wavfile.write(preview_filepath, generator.sample_rate, (beat * 32767).astype(np.int16))
    
    return send_file(
        preview_filepath,
        as_attachment=False,
        mimetype='audio/wav'
    )

def visualize_beat_to_file(beat, generator, filename=None):
    """Generate visualization of the beat and save to a file"""
    if not filename:
        filename = os.path.join(TEMP_DIR, f"visualization_{uuid.uuid4().hex[:8]}.png")
    
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(beat)) / generator.sample_rate, beat)
    plt.title('Beat Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(alpha=0.3)
    
    # Add beat markers
    for i in range(0, len(beat), generator.beat_length):
        plt.axvline(x=i / generator.sample_rate, color='r', linestyle='--', alpha=0.2)
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(beat, NFFT=1024, Fs=generator.sample_rate, noverlap=512, cmap='inferno')
    plt.title('Beat Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.ylim(0, 10000)  # Limit frequency range to focus on important parts
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return filename

@app.route('/visualize', methods=['POST'])
def visualize_beat():
    """Generate and return a visualization of the beat"""
    # Generate the beat
    beat, generator = generate_beat_from_params(request.form)
    
    # Create a BytesIO object to save the plot to
    img_data = BytesIO()
    
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(beat)) / generator.sample_rate, beat)
    plt.title('Beat Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(alpha=0.3)
    
    # Add beat markers
    for i in range(0, len(beat), generator.beat_length):
        plt.axvline(x=i / generator.sample_rate, color='r', linestyle='--', alpha=0.2)
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(beat, NFFT=1024, Fs=generator.sample_rate, noverlap=512, cmap='inferno')
    plt.title('Beat Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.ylim(0, 10000)  # Limit frequency range to focus on important parts
    
    plt.tight_layout()
    plt.savefig(img_data, format='png')
    plt.close()
    
    # Encode the image to base64
    img_data.seek(0)
    img_base64 = base64.b64encode(img_data.read()).decode('utf-8')
    
    return jsonify({
        'image': f'data:image/png;base64,{img_base64}'
    })

if __name__ == '__main__':
    app.run(debug=True)
