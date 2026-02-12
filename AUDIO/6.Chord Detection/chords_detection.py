import json
import numpy as np
import matplotlib.pyplot as plt

from essentia.standard import (
    MonoLoader,
    FrameGenerator,
    Windowing,
    Spectrum,
    SpectralPeaks,
    HPCP,
    ChordsDetection
)

# ------------------------------------------
# 1. LOAD AUDIO
# ------------------------------------------
print("Loading audio...")
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=22050)()

# ------------------------------------------
# 2. SETUP ALGORITHMS
# ------------------------------------------
window = Windowing(type="hann")
spectrum = Spectrum()
spectral_peaks = SpectralPeaks()
hpcp_algo = HPCP(size=36)

# Essentia 2.1: no parameters!
chord_detector = ChordsDetection()

frame_size = 4096
hop_size = 1024

chords = []

# ------------------------------------------
# 3. PROCESS FRAME-BY-FRAME
# ------------------------------------------
print("Detecting chords...")

for i, frame in enumerate(FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True)):
    spec = spectrum(window(frame))
    freqs, mags = spectral_peaks(spec)

    hpcp = hpcp_algo(freqs, mags)

    # IMPORTANT:
    # ChordsDetection expects a LIST of HPCP vectors → [[hpcp]]
    chord, strength = chord_detector([hpcp])

    time_sec = i * hop_size / 22050.0
    chords.append({"time": time_sec, "chord": chord, "strength": float(strength)})

# ------------------------------------------
# 4. SAVE JSON
# ------------------------------------------
with open("/data/chords_result.json", "w") as f:
    json.dump(chords, f, indent=4)

print("Saved chords_result.json")

# ------------------------------------------
# 5. PLOT TIMELINE
# ------------------------------------------
print("Plotting chord timeline...")

times = [c["time"] for c in chords]
labels = [c["chord"] for c in chords]

plt.figure(figsize=(14, 4))
plt.plot(times, [1]*len(times), alpha=0)

for t, label in zip(times, labels):
    plt.text(t, 1, label, fontsize=7, rotation=90, va='bottom')

plt.title("Chord Timeline (Essentia 2.1)")
plt.xlabel("Time (s)")
plt.yticks([])

plt.tight_layout()
plt.savefig("/data/chords_timeline.png")

print("Saved chords_timeline.png")
