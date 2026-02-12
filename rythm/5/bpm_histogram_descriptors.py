import json
import numpy as np
import matplotlib.pyplot as plt
import essentia
import essentia.standard as es

AUDIO_PATH = "/data/My_Song.wav"
JSON_OUT = "/data/bpm_histogram_descriptors.json"
PNG_OUT = "/data/bpm_histogram_descriptors.png"

print("Loading audio...")
audio = es.MonoLoader(filename=AUDIO_PATH, sampleRate=44100)()

# -------------------------------------------------------
# 1. BPM & beat intervals με RhythmExtractor2013
# -------------------------------------------------------
print("Estimating BPM and beat intervals (RhythmExtractor2013)...")

rhythm = es.RhythmExtractor2013(method="multifeature")

# ΠΡΟΣΟΧΗ στη σειρά των outputs:
# bpm, beats, beats_confidence, estimates, beats_intervals
bpm, beats, beats_conf, _, beats_intervals = rhythm(audio)

print(f"Overall BPM (RhythmExtractor2013): {bpm:.2f} bpm")
print(f"Number of beats: {len(beats)}")

# beats_intervals -> float32 essentia.array
bpm_intervals_np = np.array(beats_intervals, dtype="float32")
bpm_intervals_ess = essentia.array(bpm_intervals_np)

# -------------------------------------------------------
# 2. BPM Histogram Descriptors
# -------------------------------------------------------
print("Computing BpmHistogramDescriptors...")

bpmHistDesc = es.BpmHistogramDescriptors()

firstPeakBPM, firstPeakWeight, firstPeakSpread, \
    secondPeakBPM, secondPeakWeight, secondPeakSpread, \
    histogram = bpmHistDesc(bpm_intervals_ess)

histogram = np.array(histogram, dtype=float)

print(f"First peak BPM:  {firstPeakBPM:.2f}")
print(f"Second peak BPM: {secondPeakBPM:.2f}")

# -------------------------------------------------------
# 3. Save JSON
# -------------------------------------------------------
data = {
    "reference_bpm_rhythmExtractor": float(bpm),

    "firstPeakBPM": float(firstPeakBPM),
    "firstPeakWeight": float(firstPeakWeight),
    "firstPeakSpread": float(firstPeakSpread),

    "secondPeakBPM": float(secondPeakBPM),
    "secondPeakWeight": float(secondPeakWeight),
    "secondPeakSpread": float(secondPeakSpread),

    # index i -> i BPM
    "histogram": histogram.tolist()
}

with open(JSON_OUT, "w") as f:
    json.dump(data, f, indent=4)

print(f"Saved {JSON_OUT}")

# -------------------------------------------------------
# 4. Plot BPM histogram
# -------------------------------------------------------
print(f"Saving {PNG_OUT} ...")

bpms = np.arange(len(histogram))
max_bpm_to_show = 250

mask = bpms <= max_bpm_to_show
bpms_plot = bpms[mask]
hist_plot = histogram[mask]

plt.figure(figsize=(14, 6))
plt.bar(bpms_plot, hist_plot, width=1.0)

plt.axvline(firstPeakBPM, linestyle="--",
            label=f"1st peak: {firstPeakBPM:.1f} bpm")
if secondPeakBPM > 0:
    plt.axvline(secondPeakBPM, linestyle=":",
                label=f"2nd peak: {secondPeakBPM:.1f} bpm")

plt.title("BPM Histogram (BpmHistogramDescriptors)")
plt.xlabel("BPM")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PNG_OUT, dpi=200)

print(f"Saved {PNG_OUT}")
print("Done.")
