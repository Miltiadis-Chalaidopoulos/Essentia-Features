import numpy as np
import matplotlib.pyplot as plt
from essentia.standard import MonoLoader, PredominantPitchMelodia

# Load audio
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=44100)()

# Melodia pitch extraction
melodia = PredominantPitchMelodia(frameSize=2048,
                                  hopSize=128,
                                  guessUnvoiced=False)

pitch, confidence = melodia(audio)

# Save results as text
np.savetxt("/data/melodia_pitch.txt", pitch)
np.savetxt("/data/melodia_confidence.txt", confidence)

print("Saved melodia_pitch.txt & melodia_confidence.txt")

# ---- PLOT ----
times = np.arange(len(pitch)) * (128/44100)

plt.figure(figsize=(18,6))
plt.plot(times, pitch, '.', markersize=2)
plt.xlabel("Time (s)")
plt.ylabel("Pitch (Hz)")
plt.title("Predominant Pitch (Melodia)")
plt.grid(True)
plt.tight_layout()
plt.savefig("/data/melodia_pitch_plot.png", dpi=200)

print("Saved melodia_pitch_plot.png")
