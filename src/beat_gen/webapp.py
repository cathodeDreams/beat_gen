import os
from flask import Flask, render_template, request, send_file, url_for
import numpy as np
from scipy.io import wavfile
import tempfile
import uuid

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

@app.route('/generate', methods=['POST'])
def generate():
    # Get form parameters
    genre = request.form.get('genre', 'house')
    bpm = int(request.form.get('bpm', 128))
    bars = int(request.form.get('bars', 4))
    complexity = float(request.form.get('complexity', 0.5))
    swing = float(request.form.get('swing', 0.0))
    humanize = float(request.form.get('humanize', 0.2))
    sub_bass_type = request.form.get('sub_bass_type', 'basic')
    waveform = request.form.get('waveform', None)
    
    # Generate beat based on genre
    beat = None
    generator = None
    
    if genre == 'house':
        beat, generator = generate_house_beat(
            bpm=bpm, bars=bars, complexity=complexity, 
            swing=swing, humanize=humanize,
            sub_bass_type=sub_bass_type, waveform=waveform
        )
    elif genre == 'techno':
        beat, generator = generate_techno_beat(
            bpm=bpm, bars=bars, complexity=complexity, 
            swing=swing, humanize=humanize,
            sub_bass_type=sub_bass_type, waveform=waveform
        )
    elif genre == 'dubstep':
        beat, generator = generate_dubstep_beat(
            bpm=bpm, bars=bars, complexity=complexity, 
            swing=swing, humanize=humanize,
            sub_bass_type=sub_bass_type, waveform=waveform
        )
    elif genre == 'hardstyle':
        beat, generator = generate_hardstyle_beat(
            bpm=bpm, bars=bars, complexity=complexity, 
            swing=swing, humanize=humanize,
            sub_bass_type=sub_bass_type, waveform=waveform
        )
    else:
        # Use custom kit
        kit = get_kit_by_name(genre)
        generator = BeatGenerator(kit, bpm=bpm)
        beat = generator.generate_beat(
            bars=bars, complexity=complexity, 
            swing=swing, humanize=humanize
        )
    
    # Normalize audio
    if np.max(np.abs(beat)) > 0:
        beat = beat / np.max(np.abs(beat)) * 0.95
        
    # Generate unique filename
    filename = f"{genre}_{bpm}bpm_{uuid.uuid4().hex[:8]}.wav"
    filepath = os.path.join(TEMP_DIR, filename)
    
    # Save beat
    wavfile.write(filepath, generator.sample_rate, (beat * 32767).astype(np.int16))
    
    return send_file(
        filepath,
        as_attachment=True,
        download_name=filename,
        mimetype='audio/wav'
    )

if __name__ == '__main__':
    app.run(debug=True)