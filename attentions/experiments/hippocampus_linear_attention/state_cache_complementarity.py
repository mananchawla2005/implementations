import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

d_k = 32
d_v = 8
n_train = 300
n_heldout = 50
w = 12
n_seeds = 5
torch.manual_seed(0)


def rms_norm(x):
    return x / (x.norm(dim=-1, keepdim=True) / x.shape[-1] ** 0.5 + 1e-8)


def run(class_type, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d_k, d_v, generator=g)

    tr_keys = F.normalize(torch.randn(n_train, d_k, generator=g), dim=-1)
    tr_vals = tr_keys @ A
    exceptions = []
    if class_type == "exceptions":
        for _ in range(6):
            idx = torch.randint(0, n_train, (1,), generator=g).item()
            eps = torch.randn(d_v, generator=g) * 5.0
            tr_vals[idx] = tr_vals[idx] + eps
            exceptions.append((idx, eps))
    elif class_type == "memorize":
        tr_vals = torch.randn(n_train, d_v, generator=g)

    S = torch.zeros(d_k, d_v)
    ck, cv, cs = [], [], []
    for i in range(n_train):
        k, v = tr_keys[i], tr_vals[i]
        e = v - S.T @ k
        S = S + torch.outer(k, e)
        ck.append(k); cv.append(v); cs.append(e.norm().item())
        if len(cs) > w:
            idx = min(range(len(cs)), key=lambda j: cs[j])
            del ck[idx], cv[idx], cs[idx]

    CK = rms_norm(torch.stack(ck))
    CV = torch.stack(cv)

    def cache_out(q):
        wa = torch.softmax(rms_norm(q) @ CK.T / (d_k ** 0.5), -1)
        return wa @ CV

    te_keys = F.normalize(torch.randn(n_heldout, d_k, generator=g), dim=-1)
    if class_type == "compress":
        te_vals = te_keys @ A
    elif class_type == "memorize":
        te_vals = torch.randn(n_heldout, d_v, generator=g)
    else:
        te_vals = te_keys @ A

    heldout = {
        "state": sum((S.T @ te_keys[i] - te_vals[i]).norm().item() for i in range(n_heldout)) / n_heldout,
        "cache": sum((cache_out(te_keys[i]) - te_vals[i]).norm().item() for i in range(n_heldout)) / n_heldout,
        "combined": sum((S.T @ te_keys[i] + cache_out(te_keys[i]) - te_vals[i]).norm().item() for i in range(n_heldout)) / n_heldout,
    }

    exc = None
    if class_type == "exceptions":
        s_err = c_err = cm_err = 0.0
        for (idx, eps) in exceptions:
            q, true = tr_keys[idx], tr_vals[idx]
            s_err += (S.T @ q - true).norm().item()
            c_err += (cache_out(q) - true).norm().item()
            cm_err += (S.T @ q + cache_out(q) - true).norm().item()
        exc = {"state": s_err / len(exceptions), "cache": c_err / len(exceptions), "combined": cm_err / len(exceptions)}

    return heldout, exc


def avg(dicts):
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in dicts[0]}


for ct in ["compress", "exceptions", "memorize"]:
    hs = [run(ct, s)[0] for s in range(n_seeds)]
    a = avg(hs)
    print(f"\n=== {ct} ===")
    print(f"  held-out  state={a['state']:.3f}  cache={a['cache']:.3f}  combined={a['combined']:.3f}")
    if ct == "exceptions":
        ex = [run(ct, s)[1] for s in range(n_seeds)]
        b = avg(ex)
        print(f"  exception state={b['state']:.3f}  cache={b['cache']:.3f}  combined={b['combined']:.3f}")

comp = avg([run("compress", s)[0] for s in range(n_seeds)])
exc_h, exc_e = [run("exceptions", s)[0] for s in range(n_seeds)], [run("exceptions", s)[1] for s in range(n_seeds)]
mem = avg([run("memorize", s)[0] for s in range(n_seeds)])

assert comp["state"] < 0.2, "state should model a compressible map"
exc = avg(exc_e)
assert exc["state"] > exc["combined"], "cache should rescue exceptions"
assert mem["state"] > 1.0, "state should struggle on pure memorization"
