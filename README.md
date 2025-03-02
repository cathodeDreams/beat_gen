# Beat Generator

A Python package for generating electronic music beats in various EDM genres.

## Features

- Generate beats in multiple EDM genres (House, Techno, Dubstep, Hardstyle)
- Customize parameters like BPM, complexity, swing, and humanization
- Create different types of sub-bass sounds
- Web interface for easy beat generation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/beat-gen.git
cd beat-gen

# Install with Poetry
poetry install
```

## Usage

### Command Line

```bash
# Run the example script
poetry run python -m beat_gen.example

# Run the EDM generator demo
poetry run python -m beat_gen.demo_edm_generator

# Run the sub-bass demo
poetry run python -m beat_gen.demo_sub_bass
```

### Web Interface

```bash
# Start the web app
poetry run python -m beat_gen.run_webapp
```

Then open your browser and navigate to: http://127.0.0.1:5000

## Web Interface Features

The web interface allows you to:

1. Select the beat genre (House, Techno, Dubstep, Hardstyle)
2. Adjust BPM (beats per minute)
3. Set the number of bars to generate
4. Control complexity, swing, and humanization
5. Choose from different sub-bass types
6. Download the generated beat as a WAV file

## Example Code

```python
from beat_gen.edm_beat_generator import generate_house_beat

# Generate a house beat with custom parameters
beat, generator = generate_house_beat(
    bpm=126, 
    bars=4,
    complexity=0.7,
    swing=0.2,
    humanize=0.3,
    sub_bass_type="wobble"
)

# Save the beat
generator.save_beat(beat, "my_house_beat.wav")
```

## License

MIT