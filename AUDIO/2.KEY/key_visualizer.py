import numpy as np
import matplotlib.pyplot as plt
import essentia.standard as es

# -----------------------------------------------------
# KEY EXTRACTION
# -----------------------------------------------------
def extract_key(audio_path):
    loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
    audio = loader()

    key_extractor = es.KeyExtractor(
        profileType="edma",
        scale="all",
        tuningFrequency=440.0
    )

    key, scale, strength = key_extractor(audio)
    return audio, key, scale, strength


# -----------------------------------------------------
# VISUALIZATIONS (SAVE IMAGES)
# -----------------------------------------------------
def visualize_waveform(audio, output="waveform.png"):
    plt.figure(figsize=(14, 4))
    plt.title("Waveform")
    plt.plot(audio)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(output)
    print(f"[+] Saved waveform to {output}")


def visualize_chromagram(audio, output="chromagram.png"):
    spectrum = es.Spectrum()
    hpcp = es.HPCP()

    chroma_frames = []
    frame_gen = es.FrameGenerator(audio, frameSize=4096, hopSize=2048, startFromZero=True)

    for frame in frame_gen:
        sp = spectrum(frame)
        chroma = hpcp(sp)
        chroma_frames.append(chroma)

    chroma_matrix = np.array(chroma_frames).T

    plt.figure(figsize=(14, 6))
    plt.imshow(chroma_matrix, aspect="auto", origin="lower")
    plt.title("Chromagram (HPCP)")
    plt.xlabel("Frame")
    plt.ylabel("Pitch Class")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output)
    print(f"[+] Saved chromagram to {output}")


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    AUDIO_FILE = "audio.wav"  # put audio file next to script

    audio, key, scale, confidence = extract_key(AUDIO_FILE)

    print("\n=== KEY DETECTION RESULT ===")
    print("Key:", key)
    print("Scale:", scale)
    print("Confidence:", confidence)
    print("============================\n")

    visualize_waveform(audio)
    visualize_chromagram(audio)
