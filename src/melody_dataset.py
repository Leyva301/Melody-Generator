import os
import mido
import torch
from torch.utils.data import Dataset

class CustomMIDIDataset(Dataset):
    def __init__(self, data_dir, max_time=None, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.max_time = max_time
        self.transform = transform
        self.target_transform = target_transform

        # Collect all file paths and labels
        self.samples = []  # list of tuples (midi_path, artist_label)
        artists = sorted(os.listdir(data_dir))
        self.artist_to_idx = {artist: i for i, artist in enumerate(artists)}

        for artist in artists:
            artist_path = os.path.join(data_dir, artist)
            if not os.path.isdir(artist_path):
                continue
            for file in os.listdir(artist_path):
                if file.endswith(".mid"):
                    midi_path = os.path.join(artist_path, file)
                    self.samples.append((midi_path, artist))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        midi_path, artist = self.samples[idx]
        notes = []
        current_time = 0.0

        try:
            mid = mido.MidiFile(midi_path)
            for msg in mid:
                current_time += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    notes.append(msg.note)
                if self.max_time and current_time > self.max_time:
                    break
        except Exception as e:
            print(f"Error reading {midi_path}: {e}")
            return torch.zeros(1), torch.tensor(-1)

        sequence = torch.tensor(notes, dtype=torch.long)
        label = torch.tensor(self.artist_to_idx[artist], dtype=torch.long)

        if self.transform:
            sequence = self.transform(sequence)
        if self.target_transform:
            label = self.target_transform(label)

        return sequence, label
