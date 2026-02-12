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
    RhythmTransform
)

print("Loading audio...")
audio = MonoLoader(filename="/data/My_Song.wav", sampleRate=44100)()

# ============================================================
# PARAMETERS
# ============================================================
frame_size = 2048
hop_size = 512

window = Windowing(type="hann")
fft = FFT()
onset_det = OnsetDetection(method="flux")
rhythm_transform = RhythmTransform()

# ============================================================
# 1) COMPUTE ONSET CURVE
# ============================================================
print("Computing onset curve...")

onset_curve = []

for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
    win = window(frame)
    cpx = fft(win)
    mag = np.abs(cpx)
    phase = np.angle(cpx)

    onset_value = onset_det(mag, phase)
    onset_curve.append(onset_value)

onset_curve = np.array(onset_curve, dtype=float)

# ============================================================
# 2) FIX: OLD ESSENTIA EXPECTS VectorVectorReal FORMAT
# ============================================================
# RhythmTransform DOES NOT accept a 1D array
# It expects a LIST OF LISTS → [[val], [val], [val]...]

print("Formatting onset curve for RhythmTransform...")

onset_frames = [[float(v)] for v in onset_curve]

# ============================================================
# 3) RHYTHM TRANSFORM
# ============================================================
print("Applying RhythmTransform...")

rhythm_fingerprint = rhythm_transform(onset_frames)

# Convert to numpy for plotting + saving
rhythm_fingerprint = np.array(rhythm_fingerprint, dtype=float)

# ============================================================
# SAVE JSON
# ============================================================
output_json = "/data/rhythm_transform.json"
with open(output_json, "w") as f:
    json.dump({"rhythm_fingerprint": rhythm_fingerprint.tolist()}, f, indent=4)

print(f"Saved {output_json}")

# ============================================================
# SAVE PLOT
# ============================================================
print("Saving rhythm_transform_plot.png...")

plt.figure(figsize=(14, 6))
plt.plot(rhythm_fingerprint, linewidth=1.5)
plt.title("Rhythm Transform (Essentia)")
plt.xlabel("Modulation Frequency / Lag")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.savefig("/data/rhythm_transform_plot.png", dpi=200)

print("Saved rhythm_transform_plot.png")
