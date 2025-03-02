from beat_gen.edm_drum_library import get_kit_by_name
from beat_gen.edm_beat_generator import BeatGenerator

# Get a preset kit
kit = get_kit_by_name("house")

# Create a beat generator
generator = BeatGenerator(kit, bpm=160)

# Generate a beat
beat = generator.generate_beat(
    bars=8,
    complexity=1,
    swing=1,
    humanize=1
)

# Save the beat
generator.save_beat(beat, "my_beat.wav")

# Visualize the beat
#generator.visualize_beat(beat)
