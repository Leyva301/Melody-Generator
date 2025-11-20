import torch
import torch.nn as nn
import random
from src.melody_dataset import CustomMIDIDataset

# ---------- Load Model Definition ----------
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size=128, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out.squeeze(-1)

# ---------- Load Trained Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleLSTM().to(device)
model.load_state_dict(torch.load("melody_model.pth", map_location=device))
model.eval()

# ---------- Generate New Melody ----------
def generate_melody(seed_note=60, length=100):
    notes = [seed_note / 127.0]  # start normalized
    input_seq = torch.tensor(notes, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(length):
            pred = model(input_seq)
            temperature = 0.05  # adjust 0.01–0.2 for subtle randomness
            next_note = pred[0, -1].item() + random.uniform(-temperature, temperature)
            notes.append(max(0.0, min(1.0, next_note)))  # keep in [0,1]
            input_seq = torch.tensor(notes, dtype=torch.float32).unsqueeze(0).to(device)
    return notes

# Generate melody (you can change seed_note or length)
melody = generate_melody(seed_note=36, length=200)
melody_int = [int(n * 127) for n in melody]
print(" Generated melody (first 20 notes):", melody_int[:20])

# ---------- Export to MIDI ----------
from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for note in melody_int:
    note_on_time = random.randint(60, 160)   # random delay before note
    note_off_time = random.randint(150, 300) # random note length
    track.append(Message('note_on', note=note, velocity=64, time=note_on_time))
    track.append(Message('note_off', note=note, velocity=64, time=note_off_time))

mid.save("generated_melody.mid")
print(" Melody saved to generated_melody.mid")
