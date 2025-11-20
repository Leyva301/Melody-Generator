import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

# Import dataset class
from src.melody_dataset import CustomMIDIDataset

# ---------- Dataset Setup ----------
base_dir = "/nfs/stak/users/leyvag/ai_club/projects/Melody-Generator/Data"

def normalize_notes(seq):
    return seq.float() / 127.0

dataset = CustomMIDIDataset(data_dir=base_dir, max_time=10.0, transform=normalize_notes)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, torch.stack(labels)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)


# ---------- Model Definition ----------
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size=128, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # predicts next note (normalized float)
    
    def forward(self, x):
        x = x.unsqueeze(-1)  # add feature dim
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out.squeeze(-1)


# ---------- Training Loop ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X, _ in train_loader:
        X = X.to(device)
        # Predict next note in sequence
        y_pred = model(X[:, :-1])
        y_true = X[:, 1:]  # shift one step ahead
        loss = criterion(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "melody_model.pth")
print("Model trained and saved to melody_model.pth")