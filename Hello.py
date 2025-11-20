import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

# Add src path for import
sys.path.append("/nfs/stak/users/leyvag/ai_club/projects/Melody-Generator/src")

from melody_dataset import CustomMIDIDataset


# ---------- SETUP SECTION ----------

base_dir = "/nfs/stak/users/leyvag/ai_club/projects/Melody-Generator/Data"

# Optional: normalization transform
def normalize_notes(seq):
    return seq.float() / 127.0  # scale to [0,1]

# Create dataset object with transform
dataset = CustomMIDIDataset(data_dir=base_dir, max_time=10.0, transform=normalize_notes)
print(f"Loaded {len(dataset)} MIDI files.")


# ---------- SAMPLE TEST ----------
melody, label = dataset[0]
print("🎵 Example melody tensor:", melody[:20])
print("Artist label (integer):", label)


# ---------- DATALOADER SETUP ----------
def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, torch.stack(labels)

loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

for X, y in loader:
    print("\nBatch loaded successfully!")
    print("Batch tensor shape:", X.shape)
    print("Labels:", y)
    break


# ---------- DATA SPLIT ----------
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")



lengths = [len(dataset[i][0]) for i in range(min(200, len(dataset)))]
plt.hist(lengths, bins=30)
plt.title("MIDI Sequence Lengths")
plt.xlabel("Number of notes")
plt.ylabel("Count")
plt.show()

# Print note value range
all_notes = torch.cat([dataset[i][0] for i in range(min(200, len(dataset)))])
print(f"Note value range: {all_notes.min().item()}–{all_notes.max().item()}")
