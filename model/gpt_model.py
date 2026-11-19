import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, embed_size, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList()
        for i in range(n_head):
            self.heads.append(Head(n_embd, head_size, block_size))
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        outs = []
        for h in self.heads:
            outs.append(h(x))
        out = torch.cat(outs, dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList()
        for i in range(n_layer):
            self.blocks.append(Block(n_embd, n_head, block_size))
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

def load_model(folder, device="cpu"):
    with open(folder + "/config.json") as f:
        config = json.load(f)
    with open(folder + "/vocab.json") as f:
        chars = json.load(f)
    stoi = {}
    itos = {}
    for i, ch in enumerate(chars):
        stoi[ch] = i
        itos[i] = ch

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join([itos[i] for i in ids])

    model = GPT(config["vocab_size"], config["n_embd"], config["n_head"], config["n_layer"], config["block_size"]).to(device)
    model.load_state_dict(torch.load(folder + "/model.pt", map_location=device, weights_only=True))
    model.eval()
    return model, encode, decode, config

@torch.no_grad()
def generate_text(model, encode, decode, config, prompt, n, temperature=1.0, seed=None):
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    for _ in range(n):
        logits = model(idx[:, -config["block_size"]:])[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
    return decode(idx[0].tolist())
