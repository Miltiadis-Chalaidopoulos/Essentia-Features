import json
import numpy as np
import matplotlib.pyplot as plt
from essentia.standard import MonoLoader, BeatTrackerMultiFeature

print("Loading audio...")
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=44100)()

print("Running BeatTrackerMultiFeature...")
tracker = BeatTrackerMultiFeature()

# -------------------------------
# TRY CALLING AND AUTO-DETECT OUTPUT FORMAT
# -------------------------------
result = tracker(audio)

print("Raw return:", result)

# If it's a list → only beats are returned
if isinstance(result, (list, tuple)) and all(isinstance(x, (float, int)) for x in result):
    beats = list(map(float, result))
    confidence = []  # confidence not available

# If result is a tuple → beats + confidence
elif isinstance(result, tuple) and len(result) == 2:
    beats_raw, conf_raw = result
    beats = list(map(float, beats_raw))
    # convert confidence: may be float or list
    if isinstance(conf_raw, (list, tuple)):
        confidence = list(map(float, conf_raw))
    else:
        confidence = [float(conf_raw)]
else:
    raise ValueError(f"Unexpected BeatTrackerMultiFeature output format: {type(result)}")

print("Number of beats:", len(beats))

# Compute BPM
if len(beats) > 1:
    intervals = np.diff(beats)
    bpm = float(60.0 / np.mean(intervals))
else:
    bpm = 0.0

print("Estimated BPM:", bpm)

# Save results
data = {
    "BPM_estimated": bpm,
    "beats": beats,
    "confidence": confidence
}

with open("/data/beattracker_multifeature_results.json", "w") as f:
    json.dump(data, f, indent=4)

print("Saved beattracker_multifeature_results.json")

# -------------------------------
# PLOT SECTION
# -------------------------------
plt.figure(figsize=(16,6))

# Beats timeline
plt.subplot(2,1,1)
plt.scatter(beats, [1]*len(beats), color='blue', s=10)
plt.title(f"BeatTrackerMultiFeature Beat Timeline (Estimated BPM = {bpm:.2f})")
plt.xlabel("Time (s)")
plt.yticks([])
plt.grid(True)

# Confidence plot (if available)
plt.subplot(2,1,2)
if len(confidence) > 0:
    plt.plot(confidence, color='red')
    plt.title("Beat Confidence Over Time")
    plt.xlabel("Beat Index")
    plt.ylabel("Confidence")
else:
    plt.text(0.5, 0.5, "No confidence returned by Essentia", ha='center')
    plt.title("Beat Confidence (Not available)")
plt.grid(True)

plt.tight_layout()
plt.savefig("/data/beattracker_multifeature_plot.png", dpi=200)

print("Saved beattracker_multifeature_plot.png")
