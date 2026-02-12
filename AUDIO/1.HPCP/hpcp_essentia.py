from essentia.standard import MonoLoader, FrameGenerator, Windowing, Spectrum, SpectralPeaks, HPCP
import numpy as np
import matplotlib.pyplot as plt

audio_path = "/data/My_Song.mp3"
output_image = "/data/hpcp.png"

sr = 44100
frame_size = 4096
hop_size = 2048
hpcp_size = 36

loader = MonoLoader(filename=audio_path, sampleRate=sr)
audio = loader()

hpcp = HPCP(size=hpcp_size)
window = Windowing(type='hann', zeroPadding=0)
spectrum = Spectrum()
peaks = SpectralPeaks(sampleRate=sr, magnitudeThreshold=1e-6, minFrequency=20, maxFrequency=5000)

accum = np.zeros(hpcp_size, dtype=float)

for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
    w = window(frame)
    spec = spectrum(w)
    magFreqs, magVals = peaks(spec)
    if len(magFreqs) > 0:
        accum += hpcp(magFreqs, magVals)

if accum.sum() > 0:
    accum = accum / np.max(accum)

plt.figure(figsize=(8,4))
x = np.arange(hpcp_size)
plt.bar(x, accum)
plt.xlabel('HPCP Bin')
plt.ylabel('Normalized Strength')
plt.title('HPCP — My_Song.mp3')
plt.tight_layout()
plt.savefig(output_image)
print("Saved HPCP image to", output_image)
