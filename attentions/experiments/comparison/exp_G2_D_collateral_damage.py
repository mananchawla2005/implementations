import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import matplotlib.pyplot as plt
from algorithms.inference import (
    DeltaNet,
    GatedDeltaNet,
    KimiDeltaAttention,
    GatedDeltaAttention2,
)

d_k = 4
d_v = 3

torch.manual_seed(0)
va = torch.randn(d_v)
vb = torch.randn(d_v)

model_classes = [
    ("DeltaNet", DeltaNet),
    ("GatedDeltaNet", GatedDeltaNet),
    ("KDA", KimiDeltaAttention),
    ("GDN2", GatedDeltaAttention2),
]

overlaps = [0.0, 0.25, 0.5, 0.75, 1.0]


def make_keys(c):
    ka = torch.tensor([1.0, 0.0, 0.0, 0.0])
    kb = torch.tensor([c, (1 - c * c) ** 0.5, 0.0, 0.0])
    return ka, kb


def damage_after_full_delete(cls, ka, kb):
    m = cls(d_k, d_v)
    m.step(ka, ka, va, alpha=1.0, beta=1.0, erase_gate=1.0, write_gate=1.0)
    m.step(kb, kb, vb, alpha=1.0, beta=1.0, erase_gate=1.0, write_gate=1.0)
    base = m.step(kb, kb, torch.zeros(d_v), alpha=1.0, beta=0.0, erase_gate=1.0, write_gate=1.0)
    m.step(ka, ka, torch.zeros(d_v), alpha=1.0, beta=1.0, erase_gate=1.0, write_gate=0.0)
    after = m.step(kb, kb, torch.zeros(d_v), alpha=1.0, beta=0.0, erase_gate=1.0, write_gate=1.0)
    return (after - base).norm().item()


print("Part 1: damage to kb after FULL delete of ka vs key overlap")
print(f"{'overlap':>8}" + "".join(f"{name:>11}" for name, _ in model_classes))
damage_full = {}
for c in overlaps:
    ka, kb = make_keys(c)
    row = f"{c:>8.2f}"
    for name, cls in model_classes:
        dmg = damage_after_full_delete(cls, ka, kb)
        damage_full.setdefault(name, []).append(dmg)
        row += f"{dmg:>11.3f}"
    print(row)

# ka spans channels {0, 1}, kb spans channels {1, 2}: overlap is channel 1.
ka = torch.tensor([1.0, 1.0, 0.0, 0.0])
ka = ka / ka.norm()
kb = torch.tensor([0.0, 1.0, 1.0, 0.0])
kb = kb / kb.norm()
print(f"\nPart 2: structured keys, ka.kb = {ka @ kb:.3f} (overlap only in channel 1)")

uniform = {"b": [], "dmg": [], "resid": []}
channel = {"e": [], "dmg": [], "resid": []}


def tradeoff(delete_fn):
    m = GatedDeltaAttention2(d_k, d_v)
    m.step(ka, ka, va, alpha=1.0, erase_gate=1.0, write_gate=1.0)
    m.step(kb, kb, vb, alpha=1.0, erase_gate=1.0, write_gate=1.0)
    base = m.step(kb, kb, torch.zeros(d_v), alpha=1.0, beta=0.0, erase_gate=1.0, write_gate=1.0)
    delete_fn(m)
    after = m.step(kb, kb, torch.zeros(d_v), alpha=1.0, beta=0.0, erase_gate=1.0, write_gate=1.0)
    resid = m.step(ka, ka, torch.zeros(d_v), alpha=1.0, beta=0.0, erase_gate=1.0, write_gate=1.0)
    return (after - base).norm().item(), resid.norm().item()


# uniform erase
for b in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    dmg, resid = tradeoff(
        lambda m, b=b: m.step(ka, ka, torch.zeros(d_v), alpha=1.0, erase_gate=b, write_gate=0.0)
    )
    uniform["b"].append(b)
    uniform["dmg"].append(dmg)
    uniform["resid"].append(resid)

# per-channel erase
for e0 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    dmg, resid = tradeoff(
        lambda m, e0=e0: m.step(
            ka, ka, torch.zeros(d_v), alpha=1.0,
            erase_gate=torch.tensor([e0, 0.0, 0.0, 0.0]), write_gate=0.0,
        )
    )
    channel["e"].append(e0)
    channel["dmg"].append(dmg)
    channel["resid"].append(resid)

print(f"{'erase':>10} | {'uniform':>28} | {'per-channel (ch0 only)':>30}")
print(f"{'':>10} | {'damage':>12} {'residual':>12} | {'damage':>12} {'residual':>12}")
for i in range(len(uniform["b"])):
    print(
        f"{uniform['b'][i]:>10.2f} | "
        f"{uniform['dmg'][i]:>12.3f} {uniform['resid'][i]:>12.3f} | "
        f"{channel['dmg'][i]:>12.3f} {channel['resid'][i]:>12.3f}"
    )

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

for name, cls in model_classes:
    ax1.plot(overlaps, damage_full[name], marker="o", label=name)
ax1.set_xlabel("key overlap ka.kb")
ax1.set_ylabel("damage to kb readback")
ax1.set_title("Part 1: full delete\n(all models identical)")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

ax2.plot(uniform["resid"], uniform["dmg"], marker="o", label="uniform erase (= KDA beta)")
ax2.plot(channel["resid"], channel["dmg"], marker="s", label="GDN2 per-channel erase")
ax2.set_xlabel("residual of ka (lower = more erased)")
ax2.set_ylabel("damage to kb")
ax2.set_title("Part 2: damage/residual tradeoff")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()
