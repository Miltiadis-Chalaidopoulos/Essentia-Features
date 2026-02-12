import numpy as np
import matplotlib.pyplot as plt

from essentia.standard import (
    MonoLoader,
    FrameGenerator,
    Windowing,
    Spectrum
)

print("Loading audio...")
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=44100)()

# -------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------
frame_size = 2048
hop_size = 1024
window = Windowing(type="hann")
spectrum = Spectrum()

# -------------------------------------------------------
# FRAME PROCESSING
# -------------------------------------------------------
spectra = []

print("Computing STFT...")
for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
    spec = spectrum(window(frame))
    spectra.append(spec)

spectra = np.array(spectra)

# -------------------------------------------------------
# CUSTOM SPECTRAL NOVELTY (Foote 2000)
# -------------------------------------------------------
print("Computing Spectral Novelty Curve...")

novelty_values = []

for i in range(1, len(spectra)):
    a = spectra[i-1]
    b = spectra[i]

    # Normalize vectors
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12

    # cosine distance
    cosine_dist = 1.0 - np.dot(a, b) / (na * nb)

    novelty_values.append(cosine_dist)

novelty_values = np.array(novelty_values)

# Normalize 0–1
if np.max(novelty_values) > 0:
    novelty_values /= np.max(novelty_values)

# -------------------------------------------------------
# SAVE PLOT
# -------------------------------------------------------
t = np.arange(len(novelty_values)) * (hop_size / 44100.0)

plt.figure(figsize=(16, 5))
plt.plot(t, novelty_values, color='purple')
plt.title("Spectral Novelty Curve (Foote 2000)")
plt.xlabel("Time (s)")
plt.ylabel("Novelty")
plt.grid(True)
plt.tight_layout()
plt.savefig("/data/novelty_curve.png", dpi=200)

print("Saved novelty_curve.png")
