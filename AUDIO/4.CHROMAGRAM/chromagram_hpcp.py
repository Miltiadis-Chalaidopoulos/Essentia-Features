import numpy as np
import matplotlib.pyplot as plt

from essentia.standard import (
    MonoLoader,
    FrameGenerator,
    Windowing,
    Spectrum,
    SpectralPeaks,
    HPCP
)

# ------------------------------
# 1. Load audio
# ------------------------------
print("Loading audio...")
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=22050)()

# ------------------------------
# 2. Set up algorithms
# ------------------------------
frame_size = 4096
hop_size = 1024          # smaller hop = smoother time resolution
hpcp_size = 36           # 36 bins = 3 per semitone

window = Windowing(type="hann")
spectrum = Spectrum()
peaks = SpectralPeaks()
hpcp_algo = HPCP(size=hpcp_size)

hpcp_frames = []

# ------------------------------
# 3. Compute HPCP for each frame
# ------------------------------
print("Computing chromagram (HPCP over time)...")

for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
    spec = spectrum(window(frame))
    freqs, mags = peaks(spec)
    if len(freqs) > 0:
        hpcp_frames.append(hpcp_algo(freqs, mags))
    else:
        hpcp_frames.append(np.zeros(hpcp_size))

hpcp_frames = np.array(hpcp_frames)        # shape: (num_frames, 36)
hpcp_frames = hpcp_frames.T                # (36, num_frames) for imshow

# Normalize for visualization
hpcp_frames /= (hpcp_frames.max() + 1e-9)

# Time axis in seconds
num_frames = hpcp_frames.shape[1]
duration_sec = len(audio) / 22050.0
times = np.linspace(0, duration_sec, num_frames)

# ------------------------------
# 4. Plot chromagram
# ------------------------------
print("Plotting chromagram...")

plt.figure(figsize=(12, 6))
plt.imshow(
    hpcp_frames,
    aspect='auto',
    origin='lower',
    interpolation='nearest',
    extent=[times[0], times[-1], 0, hpcp_size]
)

plt.colorbar(label="Normalized Intensity")
plt.xlabel("Time (s)")
plt.ylabel("HPCP Bins (36 = 3 per semitone)")
plt.title("HPCP Chromagram – My_Song.wav")

plt.tight_layout()
plt.savefig("/data/chromagram_hpcp.png", dpi=150)
print("Saved chromagram to /data/chromagram_hpcp.png")
