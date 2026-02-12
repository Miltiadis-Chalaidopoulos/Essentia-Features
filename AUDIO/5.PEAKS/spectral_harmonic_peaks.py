import json
import numpy as np
import matplotlib.pyplot as plt

from essentia.standard import (
    MonoLoader,
    Windowing,
    Spectrum,
    SpectralPeaks,
    HarmonicPeaks,
    PitchYinFFT,
    FrameGenerator
)

# ------------------------------------------------
# LOAD AUDIO
# ------------------------------------------------
print("Loading audio...")
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=44100)()

# ------------------------------------------------
# INITIALIZE ALGORITHMS
# ------------------------------------------------
window = Windowing(type="hann")
spectrum = Spectrum()
spectral_peaks = SpectralPeaks(magnitudeThreshold=1e-6)
pitch_algo = PitchYinFFT()
harmonic_peaks = HarmonicPeaks()

frame_size = 4096
hop_size = 2048

spec_freqs_list = []
spec_mags_list = []
harm_freqs_list = []
harm_mags_list = []

print("Extracting spectral and harmonic peaks...")

for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):

    # Spectrum
    spec = spectrum(window(frame))

    # Raw peaks
    freqs, mags = spectral_peaks(spec)

    # Filter out zero or negative frequencies
    valid = freqs > 0
    freqs = freqs[valid]
    mags = mags[valid]

    spec_freqs_list.append(freqs)
    spec_mags_list.append(mags)

    # Estimate fundamental frequency
    f0, conf = pitch_algo(frame)

    # If no valid peaks or no fundamental -> empty harmonics
    if f0 < 1 or len(freqs) == 0:
        harm_freqs_list.append(np.zeros(0))
        harm_mags_list.append(np.zeros(0))
        continue

    # Compute harmonic peaks safely
    hf, hm = harmonic_peaks(freqs, mags, f0)
    harm_freqs_list.append(hf)
    harm_mags_list.append(hm)

# ------------------------------------------------
# PAD LISTS TO SAME LENGTH
# ------------------------------------------------
def pad_list(lst):
    maxlen = max(len(x) for x in lst)
    return np.array([
        np.pad(x, (0, maxlen - len(x))) for x in lst
    ])

spec_freqs = pad_list(spec_freqs_list)
spec_mags = pad_list(spec_mags_list)
harm_freqs = pad_list(harm_freqs_list)
harm_mags = pad_list(harm_mags_list)

# Averages
spec_freqs_avg = np.mean(spec_freqs, axis=0)
spec_mags_avg = np.mean(spec_mags, axis=0)
harm_freqs_avg = np.mean(harm_freqs, axis=0)
harm_mags_avg = np.mean(harm_mags, axis=0)

# ------------------------------------------------
# SAVE JSON
# ------------------------------------------------
result = {
    "spectral_peaks": {
        "freqs": spec_freqs_avg.tolist(),
        "mags": spec_mags_avg.tolist()
    },
    "harmonic_peaks": {
        "freqs": harm_freqs_avg.tolist(),
        "mags": harm_mags_avg.tolist()
    }
}

with open("/data/peaks_data.json", "w") as f:
    json.dump(result, f, indent=4)

print("Saved peaks_data.json")

# ------------------------------------------------
# PLOT SPECTRAL PEAKS
# ------------------------------------------------
plt.figure(figsize=(12, 5))
plt.stem(spec_freqs_avg, spec_mags_avg, basefmt=" ")
plt.title("Average Spectral Peaks")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.tight_layout()
plt.savefig("/data/spectral_peaks.png")
print("Saved spectral_peaks.png")

# ------------------------------------------------
# PLOT HARMONIC PEAKS
# ------------------------------------------------
plt.figure(figsize=(12, 5))
plt.stem(harm_freqs_avg, harm_mags_avg, basefmt=" ")
plt.title("Average Harmonic Peaks")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.tight_layout()
plt.savefig("/data/harmonic_peaks.png")
print("Saved harmonic_peaks.png")
