import json
import numpy as np
import matplotlib.pyplot as plt

from essentia.standard import (
    MonoLoader,
    RhythmExtractor2013,
    FrameGenerator,
    Windowing,
    Spectrum,
    SpectralPeaks,
    HPCP,
    ChordsDetectionBeats
)

print("Loading audio...")
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=44100)()

# Beat tracking
print("Extracting beats...")
rhythm = RhythmExtractor2013(method="multifeature")
bpm, beats, beats_conf, onset, onset_conf = rhythm(audio)

print(f"Detected BPM: {bpm:.2f}")
print(f"Detected {len(beats)} beats")

# Chord detector
chords_beats = ChordsDetectionBeats()

window = Windowing(type="hann")
spectrum = Spectrum()
peaks = SpectralPeaks()
hpcp = HPCP(size=36)

print("Computing HPCP per frame...")
frame_hpcp = []

for frame in FrameGenerator(audio, frameSize=4096, hopSize=4096):
    spec = spectrum(window(frame))
    f, m = peaks(spec)
    frame_hpcp.append(hpcp(f, m))

frame_hpcp = np.array(frame_hpcp)

print("Running ChordsDetectionBeats...")

# IMPORTANT: Only TWO arguments!
chords, strengths = chords_beats(frame_hpcp, beats)

# Save results
results = []
for t, c, s in zip(beats, chords, strengths):
    results.append({"time": float(t), "chord": c, "strength": float(s)})

with open("/data/chords_beats.json", "w") as f:
    json.dump(results, f, indent=4)

print("Saved chords_beats.json")

# Plot
print("Plotting beat-synchronized chord timeline...")

plt.figure(figsize=(18, 4))

unique = sorted(set(r["chord"] for r in results))
cmap = plt.get_cmap("tab20")
colors = {ch: cmap(i % 20) for i, ch in enumerate(unique)}

for i, r in enumerate(results):
    start = r["time"]
    end = results[i+1]["time"] if i < len(results)-1 else start + 1
    plt.barh(0.5, end-start, left=start, height=0.3,
             color=colors[r["chord"]], edgecolor="black")
    plt.text((start+end)/2, 0.1, r["chord"],
             ha="center", va="top", fontsize=8, rotation=90)

plt.yticks([])
plt.xlabel("Time (s)")
plt.title("Chord Timeline (Beat-Synchronized)")
plt.tight_layout()
plt.savefig("/data/chords_beats_timeline.png", dpi=200)

print("Saved chords_beats_timeline.png")
