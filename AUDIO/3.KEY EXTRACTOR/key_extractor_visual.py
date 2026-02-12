import json
import numpy as np
import matplotlib.pyplot as plt

from essentia.standard import (
    MonoLoader,
    KeyExtractor,
    Windowing,
    Spectrum,
    SpectralPeaks,
    HPCP,
    FrameGenerator
)

# ---------------------------------------------------------------------
# 1. LOAD AUDIO
# ---------------------------------------------------------------------
print("Loading audio...")
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=44100)()

# ---------------------------------------------------------------------
# 2. KEY EXTRACTION (FAST + ACCURATE)
# ---------------------------------------------------------------------
print("Running KeyExtractor...")
key, scale, strength = KeyExtractor(profileType="edma")(audio)

# Save JSON result
result = {"key": key, "scale": scale, "strength": strength}
with open("/data/key_extractor_result.json", "w") as f:
    json.dump(result, f, indent=4)

print("Key detected:", key, scale, "Strength:", strength)

# ---------------------------------------------------------------------
# 3. COMPUTE AVERAGE HPCP VECTOR FOR VISUALIZATION
# ---------------------------------------------------------------------
print("Computing HPCP for visualization...")

w = Windowing(type="hann")
spectrum = Spectrum()
peaks = SpectralPeaks()
hpcp_algo = HPCP(size=36)  # 36 bins = 3 bins per semitone

hpcp_accum = []

for frame in FrameGenerator(audio, frameSize=4096, hopSize=2048, startFromZero=True):
    spec = spectrum(w(frame))
    freqs, mags = peaks(spec)
    if len(freqs) > 0:
        hpcp_accum.append(hpcp_algo(freqs, mags))

hpcp_accum = np.array(hpcp_accum)
avg_hpcp = np.mean(hpcp_accum, axis=0)

# Normalize for visualization
avg_hpcp /= avg_hpcp.max()

# ---------------------------------------------------------------------
# 4. VISUALIZATION
# ---------------------------------------------------------------------
print("Creating visualization...")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# ----- BAR GRAPH -----
ax[0].bar(np.arange(len(avg_hpcp)), avg_hpcp)
ax[0].set_title(f"HPCP Profile (Key: {key} {scale})")
ax[0].set_xlabel("Pitch Class Bins")
ax[0].set_ylabel("Intensity")

# ----- CIRCULAR KEY WHEEL -----
theta = np.linspace(0, 2 * np.pi, len(avg_hpcp), endpoint=False)
ax[1] = plt.subplot(1, 2, 2, projection="polar")
ax[1].plot(theta, avg_hpcp)
ax[1].fill(theta, avg_hpcp, alpha=0.3)
ax[1].set_title("Circular Tonal Profile")

plt.tight_layout()
plt.savefig("/data/key_visualization.png")

print("Saved visualization to /data/key_visualization.png")
