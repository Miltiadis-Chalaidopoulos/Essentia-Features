from essentia.standard import MonoLoader, Key
import json

print("Starting key detection...")

audio_path = "/data/My_Song.mp3"
output_file = "/data/key_result.json"

# Load audio
loader = MonoLoader(filename=audio_path, sampleRate=44100)
audio = loader()
print("Audio loaded, length =", len(audio))

# Detect key
key_algo = Key()
key, scale, strength = key_algo(audio)

# Save results
result = {"key": key, "scale": scale, "strength": strength}

with open(output_file, "w") as f:
    json.dump(result, f, indent=4)

print("Detected key:", key, scale, "Strength:", strength)
print("Results saved to", output_file)
