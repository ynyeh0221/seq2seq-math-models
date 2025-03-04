# Required Libraries
# pip install torch numpy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Current using device：{device}")

# Special tokens
PAD, SOS, EOS = '<pad>', '<sos>', '<eos>'

# Tokenizer (digit-based tokenizer)
class Tokenizer:
    def __init__(self):
        self.vocab = [PAD, SOS, EOS, '+', '='] + [str(i) for i in range(10)]
        self.vocab2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2vocab = {idx: token for idx, token in enumerate(self.vocab)}

    def encode(self, expr):
        return [self.vocab2idx[c] for c in expr]

    def decode(self, tokens):
        return [self.idx2vocab[token.item()] for token in tokens]

# Dataset definition
class MathDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, trg = self.data[idx]
        src_tensor = torch.tensor(self.tokenizer.encode(src[::-1]), dtype=torch.long)  # Reverse input here
        trg_tensor = torch.tensor([self.tokenizer.vocab2idx[SOS]] + self.tokenizer.encode(trg) + [self.tokenizer.vocab2idx[EOS]], dtype=torch.long)
        return src_tensor, trg_tensor

# Collate function for DataLoader
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=0)
    return src_batch, trg_batch

# Function to generate addition data
def generate_addition_data(num_samples=20000, max_val=99):
    data = []
    for _ in range(num_samples):
        a, b = random.randint(1, max_val), random.randint(1, max_val)
        res = a + b
        src = f"{a}+{b}="
        trg = str(res)
        data.append((src, trg))
    return data

# Initialize tokenizer
tokenizer = Tokenizer()

train_data = generate_addition_data(20000)
test_data = generate_addition_data(500)

train_loader = DataLoader(MathDataset(train_data, tokenizer), batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(MathDataset(test_data, tokenizer), batch_size=1, shuffle=False, collate_fn=collate_fn)

# RNN (LSTM) Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, _ = self.lstm(embedded)
        predictions = self.fc_out(outputs)
        return predictions

model = RNNModel(vocab_size=len(tokenizer.vocab), embed_dim=256, hidden_dim=512, num_layers=3, dropout=0.2).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src)
        output = output[:, :trg.size(1)-1, :]
        loss = criterion(output.reshape(-1, len(tokenizer.vocab)), trg[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

# Save model
torch.save(model.state_dict(), '../addition_rnn_reverse_model.pth')

# Evaluate
def evaluate(model, loader, max_length=10):
    model.eval()
    with torch.no_grad():
        for src, trg in loader:
            src = src.to(device)
            output = model(src)
            pred_tokens = output.argmax(dim=-1)[0]
            pred_str = ''.join(tokenizer.decode(pred_tokens)).replace(PAD, '').replace(SOS, '').replace(EOS, '').strip()
            true_str = ''.join(tokenizer.decode(trg[0][1:-1])).strip()
            input_str = ''.join(tokenizer.decode(src[0])).strip()
            print(f"Input: {input_str[::-1]} Predicted: {pred_str} True: {true_str}")  # Reverse input again for clarity

print("\nRNN Test Results:")
evaluate(model, test_loader)
