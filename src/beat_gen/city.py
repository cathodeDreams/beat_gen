from beat_gen.city_pop_integration import generate_city_pop_track

# Generate a city pop track
generate_city_pop_track(
    key="A",                        # Musical key
    chord_progression_type="city_pop_1", 
    pattern_type="smooth_1",        # Bassline rhythm pattern
    bars=8,
    bpm=96,                         # City pop tends to be mid-tempo
    complexity=0.7,
    walking_bass=False,             # Set to True for walking bass jazz style
    bass_roundness=0.7,             # Tone: sine/triangle blend
    bass_brightness=0.5,            # Tone: harmonic brightness
    bass_harmonics=0.6              # Tone: harmonic content
)
