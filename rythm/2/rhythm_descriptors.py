import json
import numpy as np
import matplotlib.pyplot as plt

# 1) Load rhythm_results.json
with open("/data/rhythm_results.json", "r") as f:
    data = json.load(f)

global_bpm = float(data.get("BPM", 0.0))
beats = np.array(data.get("beats", []), dtype=float)
onsets = np.array(data.get("onsets", []), dtype=float)

if beats.size < 2:
    raise ValueError("Not enough beats to compute rhythm descriptors.")

# 2) Inter-beat intervals and instantaneous BPM
ibi = np.diff(beats)              # seconds between consecutive beats
instant_bpm = 60.0 / ibi          # BPM between beats

ibi_mean = float(ibi.mean())
ibi_std = float(ibi.std())

bpm_mean = float(instant_bpm.mean())
bpm_std = float(instant_bpm.std())

# Tempo stability (1 = perfect stable, 0 = very unstable)
tempo_stability = 1.0 - (bpm_std / bpm_mean) if bpm_mean > 0 else 0.0

# 3) Densities
duration_est = float(beats[-1] - beats[0]) if beats.size > 1 else 0.0
beat_density = float(len(beats) / duration_est) if duration_est > 0 else 0.0

if onsets.size > 0:
    onset_span = float(onsets.max() - onsets.min())
    onset_density = float(len(onsets) / onset_span) if onset_span > 0 else 0.0
else:
    onset_density = 0.0

# 4) Save descriptors to JSON
rhythm_desc = {
    "global_bpm": global_bpm,
    "bpm_mean_from_beats": bpm_mean,
    "bpm_std_from_beats": bpm_std,
    "tempo_stability": tempo_stability,
    "inter_beat_interval_mean": ibi_mean,
    "inter_beat_interval_std": ibi_std,
    "beat_density_per_second": beat_density,
    "onset_density_per_second": onset_density,
    "estimated_duration_from_beats": duration_est,
    "num_beats": int(len(beats)),
    "num_onsets": int(len(onsets))
}

with open("/data/rhythm_descriptors.json", "w") as f:
    json.dump(rhythm_desc, f, indent=4)

print("Saved rhythm_descriptors.json")

# 5) Plots
time_mid = beats[:-1] + ibi / 2.0  # time axis for instant BPM

plt.figure(figsize=(16, 9))

# (a) Histogram of instantaneous BPM
plt.subplot(3, 1, 1)
plt.hist(instant_bpm, bins=30, edgecolor="black")
plt.axvline(bpm_mean, color="red", linestyle="--", label=f"Mean = {bpm_mean:.2f} BPM")
plt.title(f"Instantaneous Tempo Histogram (global BPM = {global_bpm:.2f})")
plt.xlabel("BPM")
plt.ylabel("Count")
plt.legend()
plt.grid(True)

# (b) Tempo over time
plt.subplot(3, 1, 2)
plt.plot(time_mid, instant_bpm, marker="o", linestyle="-")
plt.axhline(global_bpm, color="red", linestyle="--", label=f"Global BPM = {global_bpm:.2f}")
plt.xlabel("Time (s)")
plt.ylabel("Instant BPM")
plt.title("Instantaneous Tempo Over Time")
plt.legend()
plt.grid(True)

# (c) Beats and onsets timeline
plt.subplot(3, 1, 3)
plt.scatter(beats, [1]*len(beats), s=10, label="Beats", color="blue")
if onsets.size > 0:
    plt.scatter(onsets, [0.5]*len(onsets), s=10, label="Onsets", color="red")
plt.yticks([0.5, 1.0], ["Onsets", "Beats"])
plt.xlabel("Time (s)")
plt.title("Beats & Onsets Timeline")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("/data/rhythm_descriptors_plot.png", dpi=200)
print("Saved rhythm_descriptors_plot.png")
