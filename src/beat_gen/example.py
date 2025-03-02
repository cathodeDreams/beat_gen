from beat_gen.edm_drum_library import get_kit_by_name
from beat_gen.edm_beat_generator import BeatGenerator

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
generator.save_beat(beat, "my_beat.wav")

# Visualize the beat
generator.visualize_beat(beat)
