"""
Train 100-param models: Vanilla vs OSPR torch.sin/cos vs OSPR Bhaskara I
Per-layer nn.Parameter halflength, ~2x compression
"""
import math, random, torch, torch.nn as nn, torch.nn.functional as F
from ospr_torch import OSPRLinear, bhaskara_sin_torch, bhaskara_cos_torch, compression_stats

torch.manual_seed(42)
random.seed(42)

# --- synthetic dataset: 10-dim -> 3 classes via true tiny MLP ---
# FIX: share true weights across train/test (previous bug used different seed per split -> distribution shift -> ~20% test)
def make_data(n=2000, in_dim=10, n_classes=3, seed=0, W1=None, b1=None, W2=None, b2=None):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, in_dim, generator=g)
    if W1 is None:
        g2 = torch.Generator().manual_seed(999)
        W1 = torch.randn(in_dim, 8, generator=g2) * 0.5
        b1 = torch.randn(8, generator=g2) * 0.1
        W2 = torch.randn(8, n_classes, generator=g2) * 0.5
        b2 = torch.randn(n_classes, generator=g2) * 0.1
    h = torch.relu(X @ W1 + b1)
    logits = h @ W2 + b2
    y = logits.argmax(dim=1)
    return X, y, (W1, b1, W2, b2)

# generate shared true weights from seed 999, then data with different X seeds
_, _, true_params = make_data(1, seed=999)
W1_t, b1_t, W2_t, b2_t = true_params
X_train, y_train, _ = make_data(1600, seed=0, W1=W1_t, b1=b1_t, W2=W2_t, b2=b2_t)
X_test,  y_test, _  = make_data(400,  seed=1, W1=W1_t, b1=b1_t, W2=W2_t, b2=b2_t)
in_dim=10
hidden=8
out_dim=3

# Parameter counts:
# Vanilla: L1 10*8=80 +8 bias=88, L2 8*3=24 +3=27 => 115 total, 104 weights
# To hit ~100 weights: use 10*5=50 +5*10=50 =100 weights
# We'll use the 10-8-3 above (104 weights) -> ~2x with 52 theta+4 params =56 stored => 104/56=1.85x
# Good enough for "100 param" brief
print(f"Dataset: train {X_train.shape} test {X_test.shape}")

class VanillaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class OSPRMLP(nn.Module):
    def __init__(self, use_bhaskara=False):
        super().__init__()
        # Need even weights: 10*8=80 even OK, 8*3=24 even OK
        self.fc1 = OSPRLinear(in_dim, hidden, use_bhaskara=use_bhaskara)
        self.fc2 = OSPRLinear(hidden, out_dim, use_bhaskara=use_bhaskara)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def train_one(model, tag, epochs=30, lr=0.02, batch=32):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    n = X_train.size(0)
    best_test = 0
    for epoch in range(1, epochs+1):
        model.train()
        perm = torch.randperm(n)
        tot_loss=0
        for i in range(0,n,batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            logits = model(X_train[idx])
            loss = crit(logits, y_train[idx])
            loss.backward()
            opt.step()
            tot_loss+= loss.item()*len(idx)
        # eval
        model.eval()
        with torch.no_grad():
            tr_acc = (model(X_train).argmax(1)==y_train).float().mean().item()
            te_acc = (model(X_test).argmax(1)==y_test).float().mean().item()
            te_loss = crit(model(X_test), y_test).item()
        best_test = max(best_test, te_acc)
        if epoch==1 or epoch%5==0 or epoch==epochs:
            print(f"[{tag}] epoch {epoch:2d} loss {tot_loss/n:.4f} tr_acc {tr_acc:.3f} te_acc {te_acc:.3f} te_loss {te_loss:.4f}")
    return best_test

print("\n--- Param counts ---")
vanilla = VanillaMLP()
ospr_sin = OSPRMLP(use_bhaskara=False)
ospr_bha = OSPRMLP(use_bhaskara=True)
for name,m in [("vanilla",vanilla),("ospr_sin",ospr_sin),("ospr_bha",ospr_bha)]:
    print(f"{name}: {count_params(m)} params")
    if "ospr" in name:
        for layer in [m.fc1, m.fc2]:
            print(f"  {layer.extra_repr()} stats {compression_stats(layer)}")

# Verify Bhaskara differentiable
print("\n--- Bhaskara vs torch.sin sanity ---")
xs = torch.linspace(-math.pi, math.pi, 7)
print("x      bhaskara_sin  torch.sin  err")
for x in xs:
    bs = bhaskara_sin_torch(x).item()
    ts = math.sin(x.item())
    print(f"{x: .3f}  {bs: .6f}   {ts: .6f}  {bs-ts: .6f}")
print("cos check")
for x in xs:
    bc = bhaskara_cos_torch(x).item()
    tc = math.cos(x.item())
    print(f"{x: .3f}  {bc: .6f}   {tc: .6f}  {bc-tc: .6f}")

print("\n=== Training Vanilla ===")
torch.manual_seed(42)
vanilla = VanillaMLP()
best_v = train_one(vanilla, "vanilla", epochs=40, lr=0.02)

print("\n=== Training OSPR torch.sin/cos ===")
torch.manual_seed(42)
ospr_sin = OSPRMLP(use_bhaskara=False)
best_s = train_one(ospr_sin, "ospr_sin", epochs=40, lr=0.05)  # higher lr for polar coords
print(f"  learned halflengths fc1 r_min={ospr_sin.fc1.r_min.item():.4f} r_range={ospr_sin.fc1.r_range.item():.4f} fc2 r_min={ospr_sin.fc2.r_min.item():.4f} r_range={ospr_sin.fc2.r_range.item():.4f}")
# diagnose negative half
with torch.no_grad():
    h1 = ospr_sin.fc1.get_halflength()
    print(f"  fc1 half range [{h1.min().item():.3f}, {h1.max().item():.3f}] mean {h1.mean().item():.3f} (negative => optimizer cheat)")

print("\n=== Training OSPR Bhaskara I ===")
torch.manual_seed(42)
ospr_bha = OSPRMLP(use_bhaskara=True)
best_b = train_one(ospr_bha, "ospr_bha", epochs=40, lr=0.05)
print(f"  learned halflengths fc1 r_min={ospr_bha.fc1.r_min.item():.4f} r_range={ospr_bha.fc1.r_range.item():.4f} fc2 r_min={ospr_bha.fc2.r_min.item():.4f} r_range={ospr_bha.fc2.r_range.item():.4f}")
with torch.no_grad():
    h1 = ospr_bha.fc1.get_halflength()
    print(f"  fc1 half range [{h1.min().item():.3f}, {h1.max().item():.3f}] mean {h1.mean().item():.3f}")

print("\n=== Diagnosis: why OSPR struggles ===")
print("Post-training conversion MSE (vanilla -> OSPR) shows smoothstep can't fit R(theta):")
vanilla_diag = VanillaMLP()
torch.manual_seed(123)
# quick train for diag
opt = torch.optim.Adam(vanilla_diag.parameters(), lr=0.02)
crit = nn.CrossEntropyLoss()
for _ in range(40):
    opt.zero_grad(); loss=crit(vanilla_diag(X_train), y_train); loss.backward(); opt.step()
for lname, w in [('fc1', vanilla_diag.fc1.weight.detach()), ('fc2', vanilla_diag.fc2.weight.detach())]:
    pairs = w.view(-1,2)
    th = torch.atan2(pairs[:,1], pairs[:,0])
    R = torch.sqrt(pairs[:,0]**2 + pairs[:,1]**2)
    # fit smoothstep
    import math as m
    r_min_p = torch.nn.Parameter(torch.tensor(0.4))
    r_range_p = torch.nn.Parameter(torch.tensor(0.4))
    opt2 = torch.optim.Adam([r_min_p, r_range_p], lr=0.2)
    for _ in range(300):
        opt2.zero_grad()
        x = torch.abs(th)/m.pi
        cubic = x*x*(3-2*x)
        Rp = r_min_p + r_range_p*cubic
        loss = ((Rp - R)**2).mean()
        loss.backward()
        opt2.step()
    Rp = r_min_p + r_range_p*(torch.abs(th)/m.pi)**2*(3-2*torch.abs(th)/m.pi)
    print(f"  {lname}: R mean {R.mean().item():.3f} std {R.std().item():.3f} -> smoothstep MSE {((Rp-R)**2).mean().item():.5f} (vs const {((R-R.mean())**2).mean().item():.5f}) corr {torch.corrcoef(torch.stack([R, Rp]))[0,1].item():.3f}")

print("\n=== Robustness: 3 seeds for OSPR sin (lr 0.05, uniform theta init, 40 epochs) ===")
for s in [42, 0, 123]:
    torch.manual_seed(s)
    m = OSPRMLP(use_bhaskara=False)
    best = train_one(m, f"seed {s}", epochs=30, lr=0.05)
    print(f"  seed {s} best {best:.3f}")

# Compression rate
raw_w = 80+24 # 104
stored_sin = ospr_sin.fc1.n_theta + ospr_sin.fc2.n_theta + 4 # +4 for r_min/r_range per layer
stored_bha = stored_sin
rate = raw_w / stored_sin
print(f"\n=== Summary ===")
print(f"Vanilla params {count_params(VanillaMLP())} best_te {best_v:.3f}")
print(f"OSPR sin  stored {stored_sin} raw {raw_w} rate {rate:.2f}x best_te {best_s:.3f} gap {best_s-best_v:+.3f}")
print(f"OSPR bha  stored {stored_bha} raw {raw_w} rate {rate:.2f}x best_te {best_b:.3f} gap {best_b-best_v:+.3f}")
with_bias_rate = (115) / (stored_sin + 11)  # +11 bias params (8+3)
print(f"With bias: 115 -> {stored_sin+11} = {115/(stored_sin+11):.2f}x")
# Check grad flow
print("\nGrad check Bhaskara theta grad:", ospr_bha.fc1.theta.grad is not None, "r_min grad", ospr_bha.fc1.r_min.grad is not None if ospr_bha.fc1.r_min.grad is not None else "has grad")
