# Beat-Gen Project Guide

## Development Commands
```bash
# Run the example script
python -m beat_gen.example

# Install project with Poetry
poetry install

# Update dependencies
poetry update

# Run specific modules
poetry run python -m beat_gen.example
poetry run python -m beat_gen.demo_edm_generator

# Run tests
poetry run pytest
```

## Code Style Guidelines
- **Imports**: Standard library first, then third-party, then local modules
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Types**: Use type hints for function parameters and return values
- **Formatting**: Follow PEP 8 guidelines (4 spaces indentation)
- **Error Handling**: Use try/except blocks with specific exceptions
- **Docstrings**: Include docstrings for modules, classes, and functions
- **Comments**: Include comments for complex algorithms and calculations
- **File Organization**: Keep related functionality in same module

## Audio Processing Conventions
- Use numpy arrays for audio data (-1.0 to 1.0 amplitude range)
- Sample rate is typically 44100Hz unless specified otherwise