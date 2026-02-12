import numpy as np
import matplotlib.pyplot as plt
import json
import essentia

from essentia.standard import (
    MonoLoader,
    FrameGenerator,
    Windowing,
    FFT,
    OnsetDetection,
    OnsetDetectionGlobal
)

print("Loading audio...")
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=44100)()

# -------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------
frame_size = 2048
hop_size = 512

window = Windowing(type="hann")
fft = FFT()

# -------------------------------------------------------
# ALGORITHMS
# -------------------------------------------------------
od_flux = OnsetDetection(method="flux")   # requires magnitude + phase
od_global = OnsetDetectionGlobal()

# -------------------------------------------------------
# COMPUTE ONSET CURVE (spectral flux)
# -------------------------------------------------------
print("Computing onset curve...")
onset_curve = []

for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
    win = window(frame)

    # Complex FFT
    cpx = fft(win)

    mag = np.abs(cpx)
    phase = np.angle(cpx)

    onset_val = od_flux(mag, phase)
    onset_curve.append(onset_val)

onset_curve = np.array(onset_curve)

# -------------------------------------------------------
# PEAK PICKING
# -------------------------------------------------------
print("Running OnsetDetectionGlobal...")

# Convert to essentia array (float32)
onset_curve_ess = essentia.array(onset_curve).astype('float32')

# Detect onsets (seconds)
onset_times = od_global(onset_curve_ess)

print("Detected onsets:", len(onset_times))

# -------------------------------------------------------
# SAVE JSON
# -------------------------------------------------------
with open("/data/onsets.json", "w") as f:
    json.dump({"onsets_sec": list(map(float, onset_times))}, f, indent=4)

print("Saved onsets.json")

# -------------------------------------------------------
# PLOT
# -------------------------------------------------------
print("Saving onset_curve_plot.png")

t = np.arange(len(onset_curve)) * (hop_size / 44100.0)

plt.figure(figsize=(16,6))
plt.plot(t, onset_curve, color='green')

for o in onset_times:
    plt.axvline(o, color='red', linestyle='--', alpha=0.5)

plt.title("Spectral Flux Onset Curve + Detected Onsets")
plt.xlabel("Time (s)")
plt.ylabel("Onset Strength")
plt.grid(True)
plt.tight_layout()
plt.savefig("/data/onset_curve_plot.png", dpi=200)

print("Saved onset_curve_plot.png")
