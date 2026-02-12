import numpy as np
import matplotlib.pyplot as plt
import json

from essentia.standard import (
    MonoLoader,
    FrameGenerator,
    Windowing,
    Loudness,
    BeatTrackerMultiFeature
)

print("Loading audio...")
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=44100)()

frame_size = 1024
hop_size = 512

print("Computing loudness envelope...")
loudness = Loudness()
loudness_vals = []

for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
    loudness_vals.append(float(loudness(frame)))

loudness_vals = np.array(loudness_vals)

print("Detecting beats...")
beats, beat_conf = BeatTrackerMultiFeature()(audio)

beat_times = np.array(beats)  # already seconds
beat_loudness = []

# Convert beat times → loudness frame index
for t in beat_times:
    frame_index = int(t * 44100 / hop_size)
    if frame_index < len(loudness_vals):
        beat_loudness.append(loudness_vals[frame_index])
    else:
        beat_loudness.append(0.0)

beat_loudness = np.array(beat_loudness)

# SAVE JSON
data = {
    "beat_times": beat_times.tolist(),
    "beat_loudness": beat_loudness.tolist(),
    "mean_loudness": float(np.mean(beat_loudness)),
    "std_loudness": float(np.std(beat_loudness))
}

with open("/data/beats_loudness.json", "w") as f:
    json.dump(data, f, indent=4)

print("Saved beats_loudness.json")

# PLOT
plt.figure(figsize=(16, 6))
plt.plot(beat_times, beat_loudness, "o-", color="magenta")
plt.title("Beats Loudness (Manual Implementation with BeatTrackerMultiFeature)")
plt.xlabel("Time (s)")
plt.ylabel("Loudness")
plt.grid(True)
plt.tight_layout()
plt.savefig("/data/beats_loudness_plot.png", dpi=200)

print("Saved beats_loudness_plot.png")
