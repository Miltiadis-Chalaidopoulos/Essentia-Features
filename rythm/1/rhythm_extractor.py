import json
import numpy as np
import matplotlib.pyplot as plt
from essentia.standard import MonoLoader, RhythmExtractor2013

print("Loading audio...")
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=44100)()

print("Running RhythmExtractor2013...")
rhythm = RhythmExtractor2013(method="multifeature")

result = rhythm(audio)

print("Raw returned values:", result)

# --- AUTO-DETECT RETURN SIGNATURE ---
if len(result) == 4:
    bpm, beats, beats_conf, onsets = result
    onsets_conf = []  # not available in this mode
elif len(result) == 5:
    bpm, beats, beats_conf, onsets, onsets_conf = result
else:
    raise ValueError(f"Unexpected return count: {len(result)}")

# --- Normalize values ---
beats = list(map(float, beats))
onsets = list(map(float, onsets))

# beats_conf may be float OR list
if isinstance(beats_conf, (list, tuple)):
    beats_conf = list(map(float, beats_conf))
else:
    beats_conf = [float(beats_conf)]  # single float → wrap in list

# onsets_conf may not exist
if isinstance(onsets_conf, (list, tuple)):
    onsets_conf = list(map(float, onsets_conf))
else:
    onsets_conf = []

print("BPM:", bpm)
print("Beats:", len(beats))
print("Onsets:", len(onsets))

# --- Save JSON ---
data = {
    "BPM": float(bpm),
    "beats": beats,
    "beats_confidence": beats_conf,
    "onsets": onsets,
    "onsets_confidence": onsets_conf
}

with open("/data/rhythm_results.json", "w") as f:
    json.dump(data, f, indent=4)

print("Saved rhythm_results.json")

# --- Plot ---
plt.figure(figsize=(18,6))
plt.scatter(beats, [1]*len(beats), color='blue', s=10, label='Beats')

if len(onsets) > 0:
    plt.scatter(onsets, [0.5]*len(onsets), color='red', s=10, label='Onsets')

plt.yticks([0.5, 1], ["Onsets", "Beats"])
plt.xlabel("Time (s)")
plt.title(f"Rhythm Timeline (BPM = {bpm:.2f})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("/data/rhythm_timeline.png", dpi=200)

print("Saved rhythm_timeline.png")
