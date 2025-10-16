"""
Load Mozart MIDI dataset from local storage.

This dataset contains multiple classical piano compositions by Mozart
in MIDI format. Each file will be read and stored into a list, and
basic metadata will be printed to the terminal, such as filename,
file size, and duration in seconds.

"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Optional: use mido to parse MIDI files for metadata
try:
    import mido
except ImportError:
    mido = None
    print("Note: 'mido' not installed. Install with 'pip install mido' for detailed info.")

plt.style.use("ggplot")

# Path to your MIDI dataset
midis_dir = "Data/albeniz"


def main():
    """Load the MIDI dataset, print summary information, and visualize durations."""

    # Resolve the absolute path and verify
    abs_path = os.path.abspath(midis_dir)
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"Directory not found: {abs_path}")

    # Collect all MIDI file paths
    midi_files = [
        os.path.join(abs_path, f)
        for f in os.listdir(abs_path)
        if f.lower().endswith((".mid", ".midi"))
    ]

    if not midi_files:
        print("No MIDI files found in directory.")
        return

    print(f"\nLoaded {len(midi_files)} MIDI files from: {abs_path}")

    # Store metadata in arrays
    filenames = []
    sizes = []
    durations = []

    for path in midi_files:
        filenames.append(os.path.basename(path))
        sizes.append(os.path.getsize(path) / 1024)  # in KB

        # Get duration if mido is installed
        if mido:
            try:
                mf = mido.MidiFile(path)
                durations.append(mf.length)
            except Exception as e:
                print(f"Warning: could not parse {path}: {e}")
                durations.append(np.nan)
        else:
            durations.append(np.nan)

    # Convert to numpy arrays for easy processing
    sizes = np.array(sizes)
    durations = np.array(durations)

    # Print dataset summary
    print("\n=== Dataset Summary ===")
    print(f"Total files: {len(filenames)}")
    print(f"Total size: {sizes.sum():.2f} KB")
    print(f"Average file size: {sizes.mean():.2f} KB")

    if not np.isnan(durations).all():
        valid_durations = durations[~np.isnan(durations)]
        print(f"Average duration: {valid_durations.mean():.2f} sec")
        print(f"Longest: {valid_durations.max():.2f} sec, Shortest: {valid_durations.min():.2f} sec")

    # Print all file details to terminal
    print("\n=== File Details ===")
    for i, name in enumerate(filenames):
        dur = "N/A" if np.isnan(durations[i]) else f"{durations[i]:.2f}s"
        print(f"{i+1:03d}. {name:<40} | {sizes[i]:6.1f} KB | {dur}")


if __name__ == "__main__":
    main()