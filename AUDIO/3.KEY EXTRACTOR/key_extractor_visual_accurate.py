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
    KeyExtractor
)

print("Loading audio...")
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=22050)()

# ---------------------------------------------------------------------
# Compute HPCP over entire track (frame-based)
# ---------------------------------------------------------------------
print("Computing HPCP over full song...")

window = Windowing(type='hann')
spectrum = Spectrum()
peaks = SpectralPeaks()
hpcp_algo = HPCP(size=36, referenceFrequency=440.0, bandPreset=False)

hpcp_frames = []

for frame in FrameGenerator(audio, frameSize=4096, hopSize=2048, startFromZero=True):
    spec = spectrum(window(frame))
    freqs, mags = peaks(spec)
    if len(freqs) > 0:
        hpcp_frames.append(hpcp_algo(freqs, mags))

hpcp_frames = np.array(hpcp_frames)
avg_hpcp = np.mean(hpcp_frames, axis=0)
avg_hpcp /= avg_hpcp.max()

# ---------------------------------------------------------------------
# Key detection using averaged HPCP
# ---------------------------------------------------------------------
print("Detecting key from HPCP...")

key, scale, strength = KeyExtractor(profileType='edma')(avg_hpcp)

print("\n=== Accurate KEYExtractor Result ===")
print("Key:", key)
print("Scale:", scale)
print("Strength:", strength)

# Save JSON result
result = {"key": key, "scale": scale, "strength": strength}
with open("/data/key_extractor_result.json", "w") as f:
    json.dump(result, f, indent=4)

# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
print("Saving visualization...")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
ax[0].bar(np.arange(len(avg_hpcp)), avg_hpcp)
ax[0].set_title(f"HPCP Profile (Detected: {key} {scale})")
ax[0].set_xlabel("Pitch Class Bins")
ax[0].set_ylabel("Intensity")

# Circular plot
theta = np.linspace(0, 2*np.pi, len(avg_hpcp), endpoint=False)
ax[1] = plt.subplot(1, 2, 2, projection="polar")
ax[1].plot(theta, avg_hpcp)
ax[1].fill(theta, avg_hpcp, alpha=0.3)
ax[1].set_title("Circular Tonal Profile")

plt.tight_layout()
plt.savefig("/data/key_visualization.png")

print("Visualization saved.")
