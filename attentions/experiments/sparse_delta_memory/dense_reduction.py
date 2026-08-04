import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from algorithms.sparse_delta_memory import sparse_delta_memory_recurrent
from algorithms.inference import GatedDeltaNet

d_k = 4
d_v = 3
N = d_k
T = 12

torch.manual_seed(0)
queries = torch.randn(T, d_k)
keys = torch.randn(T, d_k)
values = torch.randn(T, d_v)
alphas = torch.rand(T)
betas = torch.rand(T)

query_indices = torch.arange(N).unsqueeze(0).expand(T, N)
key_indices = query_indices.clone()

sdm_out, memories = sparse_delta_memory_recurrent(
    queries, query_indices, keys, key_indices, values, alphas, betas, N
)

gdn = GatedDeltaNet(d_k, d_v)
gdn_out = []
for t in range(T):
    o = gdn.step(
        queries[t], keys[t], values[t],
        alpha=alphas[t].item(), beta=betas[t].item(),
    )
    gdn_out.append(o)
gdn_out = torch.stack(gdn_out)

diff = (sdm_out - gdn_out).abs().max().item()
print(f"max |SDM_dense - GDN| = {diff:.3e}")

assert torch.allclose(sdm_out, gdn_out, atol=1e-6), "SDM dense != GDN"
