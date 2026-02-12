import json
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Load chords_result.json
# -------------------------------------------------
with open("/data/chords_result.json", "r") as f:
    chords = json.load(f)

# Ensure sorted by time
chords = sorted(chords, key=lambda c: c["time"])

times = [c["time"] for c in chords]

# Convert chord to string ALWAYS
labels = []
for c in chords:
    chord = c["chord"]
    # If chord is a list like ['C','m'], convert to "Cm"
    if isinstance(chord, list):
        chord = "".join(str(x) for x in chord)
    else:
        chord = str(chord)
    labels.append(chord)

if len(times) < 2:
    raise ValueError("Not enough chord data in chords_result.json")

# Estimate hop size (seconds) from time differences
dt = np.median(np.diff(times))

# -------------------------------------------------
# 2. Compress consecutive identical chords -> segments
# -------------------------------------------------
segments = []  # list of (start, end, chord)

current_chord = labels[0]
start_time = times[0]

for i in range(1, len(chords)):
    t = times[i]
    c = labels[i]

    if c != current_chord:
        end_time = times[i-1] + dt
        segments.append((start_time, end_time, current_chord))
        start_time = t
        current_chord = c

# Close last segment
end_time = times[-1] + dt
segments.append((start_time, end_time, current_chord))

print(f"Created {len(segments)} chord segments.")

# -------------------------------------------------
# 3. Assign a color per chord
# -------------------------------------------------
unique_chords = sorted(set(c for (_, _, c) in segments))
cmap = plt.get_cmap("tab20")
colors = {ch: cmap(i % 20) for i, ch in enumerate(unique_chords)}

# -------------------------------------------------
# 4. Plot as horizontal colored bars with text below
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(18, 3))

y_level = 0.5  # single horizontal line

for (start, end, chord) in segments:
    width = end - start
    mid = start + width / 2.0

    # Bar
    ax.barh(
        y_level,
        width,
        left=start,
        height=0.4,
        color=colors[chord],
        edgecolor="black",
        linewidth=0.3,
    )

    # Text label under the bar
    ax.text(
        mid,
        y_level - 0.35,
        chord,
        ha="center",
        va="top",
        fontsize=7,
        rotation=90,
    )

ax.set_yticks([])
ax.set_ylim(0, 1)
ax.set_xlabel("Time (s)")
ax.set_title("Chord Timeline (compressed, readable)")

ax.set_xlim(times[0], end_time)

plt.tight_layout()
plt.savefig("/data/chords_timeline_clean.png", dpi=200, bbox_inches="tight")

print("Saved /data/chords_timeline_clean.png")
