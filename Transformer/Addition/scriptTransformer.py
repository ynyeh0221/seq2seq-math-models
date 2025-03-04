# Required Libraries
# pip install torch numpy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import math

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
        src_tensor = torch.tensor(self.tokenizer.encode(src), dtype=torch.long)
        trg_tensor = torch.tensor([self.tokenizer.vocab2idx[SOS]] + self.tokenizer.encode(trg) + [self.tokenizer.vocab2idx[EOS]], dtype=torch.long)
        return src_tensor, trg_tensor

# Collate function for DataLoader
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=0)
    return src_batch, trg_batch

# Function to generate addition data
def generate_addition_data(num_samples=10000, max_val=99):
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

train_data = generate_addition_data(10000)
test_data = generate_addition_data(500)

train_loader = DataLoader(MathDataset(train_data, tokenizer), batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(MathDataset(test_data, tokenizer), batch_size=1, shuffle=False, collate_fn=collate_fn)

# Sinusoidal Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.transformer = nn.Transformer(d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, trg, trg_mask=None):
        src_emb = self.pos_encoder(self.embedding(src))
        trg_emb = self.pos_encoder(self.embedding(trg))
        output = self.transformer(src_emb, trg_emb, tgt_mask=trg_mask)
        return self.fc_out(output)

model = TransformerModel(vocab_size=len(tokenizer.vocab), embed_dim=128, num_heads=4).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        trg_input = trg[:, :-1]
        trg_mask = model.transformer.generate_square_subsequent_mask(trg_input.size(1)).to(device)
        output = model(src, trg_input, trg_mask)
        loss = criterion(output.reshape(-1, len(tokenizer.vocab)), trg[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')
torch.save(model.state_dict(), '../addition_model.pth')

# Evaluate
def evaluate(model, loader, max_length=10):
    model.eval()
    with torch.no_grad():
        for src, trg in loader:
            src = src.to(device)
            pred_seq = []
            trg_input = torch.tensor([[tokenizer.vocab2idx[SOS]]], device=device)
            for _ in range(max_length):
                trg_mask = model.transformer.generate_square_subsequent_mask(trg_input.size(1)).to(device)
                output = model(src, trg_input, trg_mask)
                pred_token = output[:, -1, :].argmax(dim=-1)
                if pred_token.item() == tokenizer.vocab2idx[EOS]:
                    break
                pred_seq.append(tokenizer.idx2vocab[pred_token.item()])
                trg_input = torch.cat([trg_input, pred_token.unsqueeze(0)], dim=1)

            pred_str = ''.join(pred_seq).strip()
            true_str = ''.join(tokenizer.decode(trg[0][1:-1])).strip()
            input_str = ''.join(tokenizer.decode(src[0])).strip()
            print(f"Input: {input_str} Predicted: {pred_str} True: {true_str}")

print("\nTransformer Test Results:")
evaluate(model, test_loader)
