import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.delta_attention import delta_attention_parallel

d = 4

key = torch.ones(d)
keys = key.unsqueeze(0).expand(2, d)
keys = F.normalize(keys, 2, 1)
key = keys[0]
queries = keys.clone()

v1 = torch.randn(d)
v2 = torch.randn(d)
values = torch.stack([v1, v2])
beta = 0.25
betas = torch.full((2,), beta)

_, states = delta_attention_parallel(queries, keys, values, betas)

pred1 = states[0] @ key
pred2 = states[1] @ key

# delta rule: S[t]@k = (1-beta) * S[t-1]@k + beta * v[t]
# at t=0: S[0] = 0, so pred0 = 0
# pred1 = (1-beta)*0 + beta*v1 = beta*v1
# pred2 = (1-beta)*pred1 + beta*v2
expected_pred1 = beta * v1
expected_pred2 = (1 - beta) * pred1 + beta * v2

print(f"v1 = [{v1}")
print(f"v2 = [{v2}]")
print(f"beta = {beta}")
print(f"\nAfter write 1 [{pred1}]")
print(f"Expected:  [{expected_pred1}]")
print(f"Match: {torch.allclose(pred1, expected_pred1, atol=1e-4)}")
print(f"\nAfter write 2 [{pred2}]")
print(f"Expected: [{expected_pred2}]")
print(f"Match: {torch.allclose(pred2, expected_pred2, atol=1e-4)}")
