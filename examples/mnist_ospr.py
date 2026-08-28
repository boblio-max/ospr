"""
Example: train a tiny OSPR MLP on synthetic data (same as train_100.py but using new API).
Run: python examples/mnist_ospr.py
"""

import torch
import torch.nn as nn
from ospr.torch import OSPRLinear, compression_stats

# synthetic data (10-dim -> 3 classes)
torch.manual_seed(42)
W1, b1 = torch.randn(10, 8) * 0.5, torch.randn(8) * 0.1
W2, b2 = torch.randn(8, 3) * 0.5, torch.randn(3) * 0.1

def make_data(n, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 10, generator=g)
    h = torch.relu(X @ W1 + b1)
    y = (h @ W2 + b2).argmax(1)
    return X, y

X_train, y_train = make_data(1600, 0)
X_test, y_test = make_data(400, 1)


class OSPRMLP(nn.Module):
    def __init__(self, use_bhaskara=False):
        super().__init__()
        self.fc1 = OSPRLinear(10, 8, use_bhaskara=use_bhaskara, halflength_mode="smoothstep_pos")
        self.fc2 = OSPRLinear(8, 3, use_bhaskara=use_bhaskara, halflength_mode="smoothstep_pos")

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


for name, bh in [("OSPR sin", False), ("OSPR bhaskara", True)]:
    torch.manual_seed(42)
    m = OSPRMLP(use_bhaskara=bh)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    crit = nn.CrossEntropyLoss()
    for epoch in range(20):
        opt.zero_grad()
        loss = crit(m(X_train), y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (m(X_test).argmax(1) == y_test).float().mean().item()
    print(f"{name}: test acc {acc:.3f}  {compression_stats(m.fc1)}")

print("done")
