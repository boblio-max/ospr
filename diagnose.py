import torch, math, torch.nn as nn, torch.nn.functional as F
from train_100 import X_train, y_train, X_test, y_test
from ospr_torch import OSPRLinear

def run_one(seed, mode, bh, lr=0.02, epochs=30):
    torch.manual_seed(seed)
    class OSPRMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1=OSPRLinear(10,8,use_bhaskara=bh, halflength_mode=mode)
            self.fc2=OSPRLinear(8,3,use_bhaskara=bh, halflength_mode=mode)
        def forward(self,x):
            return self.fc2(torch.relu(self.fc1(x)))
    m=OSPRMLP()
    opt=torch.optim.Adam(m.parameters(), lr=lr)
    crit=nn.CrossEntropyLoss()
    best=0
    for epoch in range(epochs):
        opt.zero_grad()
        loss=crit(m(X_train), y_train)
        loss.backward()
        opt.step()
        with torch.no_grad():
            te=(m(X_test).argmax(1)==y_test).float().mean().item()
            if te>best:
                best=te
    return best

print('Multiple seeds, smoothstep, bh=False, lr 0.02:')
for s in [42,0,1,123,999]:
    b=run_one(s, 'smoothstep', False)
    print(f'seed {s}: {b:.3f}')
print('Multiple seeds, smoothstep_pos:')
for s in [42,0,1,123]:
    print(f'seed {s}: {run_one(s, "smoothstep_pos", False):.3f}')
print('Multiple seeds, free_R:')
for s in [42,0,1,123]:
    print(f'seed {s}: {run_one(s, "free", False):.3f}')
print()
print('LR sensitivity smoothstep seed 42:')
for lr in [0.005, 0.01, 0.02, 0.05]:
    print(f'lr {lr}: {run_one(42, "smoothstep", False, lr=lr):.3f}')

class Vanilla(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(10,8)
        self.fc2=nn.Linear(8,3)
    def forward(self,x):
        return self.fc2(torch.relu(self.fc1(x)))
def run_vanilla(seed, lr=0.02):
    torch.manual_seed(seed)
    v=Vanilla()
    opt=torch.optim.Adam(v.parameters(), lr=lr)
    crit=nn.CrossEntropyLoss()
    best=0
    for _ in range(30):
        opt.zero_grad()
        loss=crit(v(X_train), y_train)
        loss.backward()
        opt.step()
        with torch.no_grad():
            te=(v(X_test).argmax(1)==y_test).float().mean().item()
            if te>best:
                best=te
    return best
print()
print('Vanilla multiple seeds:')
for s in [42,0,1,123]:
    print(f'seed {s}: {run_vanilla(s):.3f}')
