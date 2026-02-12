import json
import os
import numpy as np

# Προσπαθούμε να φορτώσουμε Essentia (αν τρέχεις μέσα σε official docker image θα δουλέψει)
try:
    from essentia.standard import ChordsDescriptors
    HAVE_ESSENTIA = True
except ImportError:
    HAVE_ESSENTIA = False

# Προσπαθούμε να φορτώσουμε matplotlib για τα plots
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# -------------------------------------------------------------------
# 1. Φόρτωση δεδομένων από /data/chords_beats.json
# -------------------------------------------------------------------

INPUT_PATH = "/data/chords_beats.json"
OUTPUT_JSON = "/data/chords_descriptors.json"
OUTPUT_PNG = "/data/chords_descriptors_plot.png"

print(f"Loading beat-synchronized chords from: {INPUT_PATH}")

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο {INPUT_PATH}. "
                            f"Σιγουρέψου ότι το έχεις κάνει mount με -v στο docker.")

with open(INPUT_PATH, "r") as f:
    data = json.load(f)

if len(data) == 0:
    raise ValueError("Το chords_beats.json είναι άδειο.")

times = np.array([d["time"] for d in data], dtype=float)
chords = [str(d["chord"]) for d in data]
strengths = np.array([d["strength"] for d in data], dtype=float)

n = len(chords)
print(f"Loaded {n} chord events.")

# -------------------------------------------------------------------
# 2. Υπολογισμός βασικών "χειροποίητων" descriptors (χωρίς Essentia)
# -------------------------------------------------------------------

# Δείκτης αλλαγής συγχορδίας: 1 όταν αλλάζει η συγχορδία, 0 αλλιώς
chord_change = np.zeros(n, dtype=float)
if n > 1:
    prev = np.array(chords[:-1], dtype=object)
    curr = np.array(chords[1:], dtype=object)
    chord_change[1:] = (curr != prev).astype(float)

# Συχνότητα εμφάνισης κάθε συγχορδίας
unique_chords, chord_counts = np.unique(chords, return_counts=True)
chord_histogram = dict(zip(unique_chords.tolist(), chord_counts.astype(int).tolist()))

# Ποσοστό αλλαγών συγχορδιών στο σύνολο
chord_change_rate = float(chord_change.sum() / (n - 1)) if n > 1 else 0.0

# -------------------------------------------------------------------
# 3. Προαιρετικά: Χρήση Essentia ChordsDescriptors (αν υπάρχει)
# -------------------------------------------------------------------

essentia_result = None

if HAVE_ESSENTIA:
    print("Essentia found: running ChordsDescriptors...")

    # ΠΡΟΣΟΧΗ: εδώ πρέπει εσύ να αποφασίσεις/global key & scale.
    # Για παράδειγμα βάζω D major. Άλλαξέ το αν ξέρεις την τονικότητα του κομματιού.
    global_key = "D"
    global_scale = "major"  # ή "minor"

    # Δημιουργία αλγορίθμου
    cd = ChordsDescriptors()

    # Σύμφωνα με το API της Essentia 2.1, η κλήση είναι:
    # chordsHistogram (vector_real),
    # chordsNumberRate (real),
    # chordsChangesRate (real),
    # chordsKey (string),
    # chordsScale (string)
    ch_hist_vec, ch_number_rate, ch_changes_rate, ch_key, ch_scale = cd(
        chords, global_key, global_scale
    )

    essentia_result = {
        "inputKey": global_key,
        "inputScale": global_scale,
        "chordsKey": ch_key,
        "chordsScale": ch_scale,
        "chordsNumberRate": float(ch_number_rate),
        "chordsChangesRate": float(ch_changes_rate),
        "chordsHistogramVector": [float(x) for x in ch_hist_vec],
    }

else:
    print("Essentia ΔΕΝ βρέθηκε – θα γραφτούν μόνο basic descriptors από τα δεδομένα JSON.")

# -------------------------------------------------------------------
# 4. Αποθήκευση όλων των descriptors σε JSON
# -------------------------------------------------------------------

result = {
    "numEvents": int(n),
    "uniqueChords": unique_chords.tolist(),
    "chordHistogram": chord_histogram,
    "chordChangeRate": chord_change_rate,
    "haveEssentia": HAVE_ESSENTIA,
}

if essentia_result is not None:
    result["essentiaDescriptors"] = essentia_result

with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=4)

print(f"Saved descriptors to: {OUTPUT_JSON}")

# -------------------------------------------------------------------
# 5. Προαιρετικά plots (αν υπάρχει matplotlib)
# -------------------------------------------------------------------

if HAVE_MPL:
    print(f"Creating plot: {OUTPUT_PNG}")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    # (α) Strength ανά χρόνο
    axes[0].plot(times, strengths)
    axes[0].set_ylabel("Strength")
    axes[0].grid(True)

    # (β) Δείκτης αλλαγής συγχορδίας
    axes[1].step(times, chord_change, where="post")
    axes[1].set_ylabel("Chord change (0/1)")
    axes[1].grid(True)

    # (γ) Bar chart με συχνότητες συγχορδιών
    chord_labels = list(chord_histogram.keys())
    chord_values = list(chord_histogram.values())
    axes[2].bar(chord_labels, chord_values)
    axes[2].set_ylabel("Count")
    axes[2].set_xlabel("Chord")
    axes[2].grid(True, axis="y")
    plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    print(f"Saved plot to: {OUTPUT_PNG}")
else:
    print("matplotlib ΔΕΝ βρέθηκε – δεν θα δημιουργηθεί PNG plot.")

print("Done.")
