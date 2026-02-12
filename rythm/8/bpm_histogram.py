import json
import numpy as np
import matplotlib.pyplot as plt
import essentia
import essentia.standard as es

AUDIO_PATH = "/data/My_Song.wav"
JSON_OUT = "/data/bpm_histogram.json"
PNG_OUT = "/data/bpm_histogram.png"

print("Loading audio...")
audio = es.MonoLoader(filename=AUDIO_PATH, sampleRate=44100)()

# -------------------------------------------------------
# 1. BPM reference με RhythmExtractor2013
#    (δεν είναι descriptor, απλώς για βαθμονόμηση)
# -------------------------------------------------------
print("Estimating reference BPM with RhythmExtractor2013...")
rhythm = es.RhythmExtractor2013(method="multifeature")
bpm_ref, beats, beats_conf, bpm_intervals, bpm_intervals_conf = rhythm(audio)
print(f"Reference BPM (RhythmExtractor2013): {bpm_ref:.2f}")

# -------------------------------------------------------
# 2. Novelty curve για BpmHistogram
# -------------------------------------------------------
frame_size = 2048
hop_size = 512
sample_rate = 44100.0

window = es.Windowing(type="hann")
spectrum = es.Spectrum()
freqBands = es.FrequencyBands()
noveltyAlgo = es.NoveltyCurve()

print("Computing frequency bands for NoveltyCurve...")

bands_list = []
for frame in es.FrameGenerator(audio,
                               frameSize=frame_size,
                               hopSize=hop_size,
                               startFromZero=True):
    spec = spectrum(window(frame))
    bands = freqBands(spec)
    bands_list.append(bands)

bands_mat = np.array(bands_list, dtype="float32")
novelty = noveltyAlgo(essentia.array(bands_mat))

# -------------------------------------------------------
# 3. BpmHistogram στο εύρος γύρω από bpm_ref
# -------------------------------------------------------
frame_rate_novelty = sample_rate / hop_size

minBpm = max(40.0, bpm_ref * 0.5)
maxBpm = min(220.0, bpm_ref * 1.8)

print(f"Running BpmHistogram in range [{minBpm:.1f}, {maxBpm:.1f}] BPM...")

bpmHist = es.BpmHistogram(
    frameRate=frame_rate_novelty,
    minBpm=minBpm,
    maxBpm=maxBpm,
    constantTempo=False,
)

bpm_mean_raw, bpmCandidates_raw, bpmMagnitudes_raw, tempogram, \
    frameBpms_raw, ticks, ticksMagnitude, sinusoid = bpmHist(novelty)

bpmCandidates_raw = np.array(bpmCandidates_raw, dtype=float)
bpmMagnitudes_raw = np.array(bpmMagnitudes_raw, dtype=float)
frameBpms_raw = np.array(frameBpms_raw, dtype=float)

print(f"Raw mean BPM (BpmHistogram): {bpm_mean_raw:.2f}")

# -------------------------------------------------------
# 4. Histogram από τα frameBpms (raw)
# -------------------------------------------------------
print("Building raw BPM histogram from frameBpms...")

valid = (frameBpms_raw >= minBpm) & (frameBpms_raw <= maxBpm)
frameBpms_valid = frameBpms_raw[valid]

if len(frameBpms_valid) == 0:
    print("WARNING: no valid frameBpms, fallback to single value.")
    frameBpms_valid = np.array([bpm_mean_raw])

bins_raw = np.arange(int(minBpm), int(maxBpm) + 1)
hist_counts_raw, bin_edges_raw = np.histogram(frameBpms_valid, bins=bins_raw)

bpm_bins_raw = bin_edges_raw[:-1]
peak_idx_raw = int(np.argmax(hist_counts_raw))
peak_bpm_raw = float(bpm_bins_raw[peak_idx_raw])

print(f"Peak BPM (raw hist): {peak_bpm_raw:.2f}")

# -------------------------------------------------------
# 5. Scaling του histogram ώστε ο μέσος BpmHistogram
#    να ευθυγραμμιστεί με το bpm_ref
# -------------------------------------------------------
if bpm_mean_raw > 0:
    scale_factor = bpm_ref / bpm_mean_raw
else:
    scale_factor = 1.0

print(f"Scale factor = bpm_ref / bpm_mean_raw = {scale_factor:.3f}")

bpm_bins_scaled = bpm_bins_raw * scale_factor
peak_bpm_scaled = peak_bpm_raw * scale_factor
bpm_mean_scaled = bpm_mean_raw * scale_factor

print(f"Scaled mean BPM (hist): {bpm_mean_scaled:.2f}")
print(f"Scaled peak BPM (hist): {peak_bpm_scaled:.2f}")

# -------------------------------------------------------
# 6. Save JSON
# -------------------------------------------------------
data = {
    "reference_bpm_rhythmExtractor": float(bpm_ref),

    "bpm_mean_raw": float(bpm_mean_raw),
    "peak_bpm_raw": float(peak_bpm_raw),
    "bpm_bins_raw": bpm_bins_raw.tolist(),
    "hist_counts_raw": hist_counts_raw.tolist(),

    "scale_factor": float(scale_factor),
    "bpm_mean_scaled": float(bpm_mean_scaled),
    "peak_bpm_scaled": float(peak_bpm_scaled),
    "bpm_bins_scaled": bpm_bins_scaled.tolist(),
}

with open(JSON_OUT, "w") as f:
    json.dump(data, f, indent=4)

print(f"Saved {JSON_OUT}")

# -------------------------------------------------------
# 7. Plot με τον ΣCALEΔ άξονα BPM
# -------------------------------------------------------
print(f"Saving {PNG_OUT} ...")

plt.figure(figsize=(14, 6))
plt.bar(bpm_bins_scaled, hist_counts_raw, width=1.0)
plt.axvline(peak_bpm_scaled, linestyle="--",
            label=f"Peak BPM (scaled hist): {peak_bpm_scaled:.1f}")
plt.title("BPM Histogram (from BpmHistogram frameBpms, scaled)")
plt.xlabel("BPM")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PNG_OUT, dpi=200)

print(f"Saved {PNG_OUT}")
