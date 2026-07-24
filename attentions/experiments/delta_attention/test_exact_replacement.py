import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from algorithms.delta_attention import delta_attention_parallel

d = 4

key = torch.ones(d) / (d ** 0.5)
keys = key.unsqueeze(0).expand(2, d)
keys = F.normalize(keys, 2, 1)
queries = keys.clone()

v1 = torch.randn(d)
v2 = torch.randn(d)
values = torch.stack([v1, v2])
betas = torch.ones(2)

_, states = delta_attention_parallel(queries, keys, values, betas)

read0 = states[0] @ key
read1 = states[1] @ key

print(f"v1 = [{', '.join(f'{x:.4f}' for x in v1)}]")
print(f"After write 1: read back = [{', '.join(f'{x:.4f}' for x in read0)}]")
print(f"Match v1: {torch.allclose(read0, v1, atol=1e-4)}")
print()
print(f"v2 = [{', '.join(f'{x:.4f}' for x in v2)}]")
print(f"After write 2: read back = [{', '.join(f'{x:.4f}' for x in read1)}]")
print(f"Match v2: {torch.allclose(read1, v2, atol=1e-4)}")
