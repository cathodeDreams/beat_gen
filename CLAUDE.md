# Beat-Gen Project Guide

## Development Commands
```bash
# Install project with Poetry
poetry install

# Run example scripts
poetry run python -m beat_gen.example
poetry run python -m beat_gen.demo_edm_generator
poetry run python -m beat_gen.demo_advanced_fills
poetry run python -m beat_gen.demo_polyrhythms

# Generate specific beat types
poetry run python -c "from beat_gen.edm_beat_generator import generate_house_beat; beat, gen = generate_house_beat(bpm=126); gen.save_beat(beat, 'output/house_beat.wav')"

# Run tests (future - no tests currently exist)
poetry run pytest
poetry run pytest tests/test_file.py::test_function  # Run specific test when added

# Update dependencies
poetry update
```

## Code Style Guidelines
- **Imports**: Standard library first, then third-party (numpy, scipy), then local modules
- **Naming**: snake_case for variables/methods, CamelCase for classes, UPPER_CASE for constants
- **Types**: Add type hints for function params and return values (numpy.ndarray, float, etc.)
- **Formatting**: PEP 8 (4 spaces indent, 88 char line limit, blank lines between functions)
- **Error Handling**: Use try/except with specific exceptions (ValueError, TypeError)
- **Docstrings**: Required for all public functions/classes, following Google style format
- **Comments**: Required for complex algorithms, especially DSP and audio calculations

## Audio Processing Conventions
- Audio data: numpy arrays with -1.0 to 1.0 amplitude range
- Sample rate: 44100Hz default
- Time signatures: Specified as tuple (beats_per_bar, beat_value)
- Beat organization: Follow musical convention (bars, beats, 16th notes)