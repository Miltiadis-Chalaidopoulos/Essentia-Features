import numpy as np
import matplotlib.pyplot as plt
from essentia.standard import (
    MonoLoader, 
    Windowing, 
    Spectrum, 
    FrameGenerator,
    SpectralPeaks,
    TuningFrequency
)

print("Loading audio...")
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=44100)()

window = Windowing(type='hann')
spectrum = Spectrum()
peaks = SpectralPeaks()
tuning = TuningFrequency()

tuning_hz_list = []
tuning_cents_list = []

print("Estimating tuning frequency...")

for frame in FrameGenerator(audio, frameSize=4096, hopSize=2048):
    mag_spectrum = spectrum(window(frame))

    # Extract peaks
    peak_freqs, peak_mags = peaks(mag_spectrum)

    if len(peak_freqs) == 0:
        continue

    # TuningFrequency returns: (tuningHz, tuningCents)
    tuningHz, tuningCents = tuning(peak_freqs, peak_mags)

    if tuningHz > 0:
        tuning_hz_list.append(tuningHz)
        tuning_cents_list.append(tuningCents)

tuning_hz_list = np.array(tuning_hz_list)
tuning_cents_list = np.array(tuning_cents_list)

if len(tuning_hz_list) == 0:
    print("ERROR: No tuning could be estimated.")
    exit()

mean_hz = tuning_hz_list.mean()
mean_cents = tuning_cents_list.mean()

print("Estimated tuning (Hz):", mean_hz)
print("Cents deviation:", mean_cents)

with open("/data/tuning_results.txt", "w") as f:
    f.write(f"Mean Tuning Frequency: {mean_hz} Hz\n")
    f.write(f"Mean Cents Offset: {mean_cents} cents\n")

# PLOT histogram of tuning frequencies
plt.figure(figsize=(14, 6))
plt.hist(tuning_hz_list, bins=40, color='skyblue', edgecolor='black')
plt.axvline(mean_hz, color='red', linestyle='--', label=f"Mean = {mean_hz:.2f} Hz")
plt.title("Tuning Frequency Histogram")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("/data/tuning_frequency_plot.png", dpi=200)

print("Saved tuning_results.txt and tuning_frequency_plot.png")
