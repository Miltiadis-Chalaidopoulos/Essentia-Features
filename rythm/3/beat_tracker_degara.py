import json
import numpy as np
import matplotlib.pyplot as plt

from essentia.standard import MonoLoader, BeatTrackerDegara

print("Loading audio...")
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=44100)()

print("Running BeatTrackerDegara...")
tracker = BeatTrackerDegara()

beats = tracker(audio)  # only returns a list of beat times

beats = list(map(float, beats))
print("Number of beats:", len(beats))

# Compute BPM from beat intervals
if len(beats) > 1:
    intervals = np.diff(beats)
    bpm = float(60.0 / np.mean(intervals))
else:
    bpm = 0.0

print("Estimated BPM:", bpm)

# Save results
data = {
    "BPM_estimated": bpm,
    "beats": beats
}

with open("/data/beattracker_degara_results.json", "w") as f:
    json.dump(data, f, indent=4)

print("Saved beattracker_degara_results.json")

# Plot
plt.figure(figsize=(16,4))
plt.scatter(beats, [1]*len(beats), color='blue', s=10)
plt.title(f"BeatTrackerDegara Beat Timeline (Estimated BPM = {bpm:.2f})")
plt.xlabel("Time (s)")
plt.yticks([])
plt.grid(True)
plt.tight_layout()
plt.savefig("/data/beattracker_degara_plot.png", dpi=200)

print("Saved beattracker_degara_plot.png")
