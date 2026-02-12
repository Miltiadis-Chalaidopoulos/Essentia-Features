from essentia.standard import MonoLoader, Key

print("Starting fast key detection...")

# Load audio at lower sample rate for speed
loader = MonoLoader(filename='/data/My_Song.mp3', sampleRate=22050)
audio = loader()
print("Audio loaded, length =", len(audio))

# Detect key
key_algo = Key()
key, scale, strength = key_algo(audio)

print("Detected key:", key, scale, "Strength:", strength)
