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
        self.vocab = [PAD, SOS, EOS, '-', '='] + [str(i) for i in range(10)]
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
        src_tensor = torch.tensor(self.tokenizer.encode(src[::-1]), dtype=torch.long)
        trg_tensor = torch.tensor([self.tokenizer.vocab2idx[SOS]] + self.tokenizer.encode(trg) + [self.tokenizer.vocab2idx[EOS]], dtype=torch.long)
        return src_tensor, trg_tensor

# Collate function for DataLoader
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=0)
    return src_batch, trg_batch

# Function to generate subtraction data
def generate_subtraction_data(num_samples=50000, max_val=999):
    data = []
    for _ in range(num_samples):
        a, b = random.randint(1, max_val), random.randint(1, max_val)
        if a < b:
            a, b = b, a  # Ensure positive results
        res = a - b
        src = f"{a}-{b}="
        trg = str(res)
        data.append((src, trg))
    return data

# Initialize tokenizer
tokenizer = Tokenizer()

train_data = generate_subtraction_data(50000)
test_data = generate_subtraction_data(500)

train_loader = DataLoader(MathDataset(train_data, tokenizer), batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(MathDataset(test_data, tokenizer), batch_size=1, shuffle=False, collate_fn=collate_fn)

# Encoder-Decoder Model
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

encoder = Encoder(len(tokenizer.vocab), embed_dim=128, hidden_dim=256).to(device)
decoder = Decoder(len(tokenizer.vocab), embed_dim=128, hidden_dim=256).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0005)

num_epochs = 50
teacher_forcing_ratio = 0.5
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0
    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        hidden = encoder(src)
        input = trg[:, 0]
        loss = 0

        for t in range(1, trg.size(1)):
            output, hidden = decoder(input, hidden)
            loss += criterion(output, trg[:, t])
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() / trg.size(1)

    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

# Evaluate
def evaluate(encoder, decoder, loader, max_length=10):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for src, trg in loader:
            src = src.to(device)
            hidden = encoder(src)
            input = torch.tensor([tokenizer.vocab2idx[SOS]], device=device)
            pred_seq = []

            for _ in range(max_length):
                output, hidden = decoder(input, hidden)
                pred_token = output.argmax(1).item()
                if pred_token == tokenizer.vocab2idx[EOS]:
                    break
                pred_seq.append(tokenizer.idx2vocab[pred_token])
                input = torch.tensor([pred_token], device=device)

            input_str = ''.join(tokenizer.decode(src[0])).strip()[::-1]
            pred_str = ''.join(pred_seq)
            true_str = ''.join(tokenizer.decode(trg[0][1:-1])).strip()
            print(f"Input: {input_str} Predicted: {pred_str} True: {true_str}")

print("\nRNN Test Results:")
evaluate(encoder, decoder, test_loader)
