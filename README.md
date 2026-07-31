# Implementations

A practical exploration of different architectures and research directions across pretraining and postraining of LLMs.

## Project Structure

```
attentions/
  algorithms/         # Core implementations (parallel + chunked + recurrent)
  experiments/        # Tests and benchmarks for understanding each algorithm
    shared/           # Plotting and benchmarking utilities
    comparison/       # Cross-model benchmark suite (Stage 5)
    linear_attention/
    softmax_attention/
    gated_linear_attention/
    delta_attention/
    gated_delta_attention/
    kimi_delta_attention/
    gated_delta_attention2/
```

## Algorithms

### Linear Attention

Standard linear attention with state = sum of outer products.

    S[t] = S[t-1] + outer(v[t], k[t])

- **Recurrent**: O(T) sequential, O(1) memory per step
- **Parallel**: O(T) via cumulative sum, fully parallelizable

**Key property**: State grows unbounded (no forgetting).

### Gated Linear Attention

Adds a scalar gate alpha[t] per timestep that decays the state before writing:

    S[t] = alpha[t] * S[t-1] + outer(v[t], k[t])

Convention: **decay, then write, then read**.

Uses parallel cumulative-product formulation for O(T) computation. Falls back to recurrent when alpha contains exact zeros.

### Delta Attention

Only writes the residual — the difference between the new value and the current state's prediction:

    S[t] = S[t-1] + beta[t] * outer(v[t] - S[t-1] @ k[t], k[t])

- Replaces values for repeated keys instead of accumulating
- Uses associative prefix scan for O(T) parallel computation
- Includes NLMS variant (not generally used — not scale invariant)

### Gated DeltaNet

Combines gating with the delta rule. The state is decayed before computing both the prediction and the update:

    S[t] = alpha[t] * S[t-1] + beta[t] * outer(v[t] - alpha[t] * (S[t-1] @ k[t]), k[t])

The two alphas cancel when writing to an existing key (exact replacement regardless of alpha). The gate only matters when no write happens (pure decay) or when reading with a different key (interference direction).

**Chunked variants** (both DeltaNet and Gated DeltaNet):
Process a block of tokens using a triangular solve within each chunk, then update a checkpoint state between chunks. This gives O(C²) per chunk with O(log C) parallel depth internally, while keeping the cross-chunk recurrence sequential.

### Kimi Delta Attention

Per-channel decay with the delta rule. Unlike Gated DeltaNet's scalar gate, the state is decayed elementwise along the key dimension with a per-channel alpha:

    S[t] = alpha[t] ⊙ S[t-1] + beta[t] * outer(k[t], v[t] - (alpha[t] ⊙ S[t-1])ᵀ k[t])

State layout is [d_k, d_v] (transposed relative to the other models); reads use Sᵀ q and writes use outer(k, error).

**Key property**: independent forgetting per key channel. Because the decay is elementwise, each channel can forget at its own rate — a single channel can be reset (alpha=0) while the rest persist, enabling anisotropic (directional) forgetting rather than a global scalar gate.

### Gated DeltaNet-2 (erase/write gates)

Two independent gates decouple "how much old content to erase" from "how much new content to write":

    decayed   = alpha[t] ⊙ S[t-1]
    e_t       = erase_gate[t] ⊙ k[t]      # per-channel gate on the key (prediction/erase side)
    z_t       = write_gate[t] ⊙ v[t]      # per-channel gate on the value (write side)
    error     = z_t - decayedᵀ e_t
    S[t]      = decayed + outer(k[t], error)

- **erase_gate** (per d_k): gates the key used for prediction, so it controls how much of the old association is subtracted (b in the memory-op table).
- **write_gate** (per d_v): gates the value committed, so it controls how much new content is written (w in the memory-op table).

The two scalar gates span a full 2-D control surface — insert (0,1), replace (1,1), delete (1,0), ignore (0,0) — where models with a single scalar gate (Gated DeltaNet, KDA) are limited to the b=w diagonal.

## Inference Interface

Located in `algorithms/inference.py`. A step-by-step interface wrapping each model behind a common API:

```python
class MemoryModel:
    def reset(self): ...
    def step(self, query, key, value, **controls): ...
```

Controls let you pass alpha and beta per step. While the parallel algorithms in `algorithms/` are for training (process all T tokens at once via prefix scan), the step-by-step interface is for autoregressive inference with O(1) memory, O(1) compute per token.

## Experiments

Run from the `attentions/` directory:

```bash
python experiments/<algorithm>/<test_name>.py
```

### Linear Attention
| File | Description |
|------|-------------|
| `repeated_key.py` | Write same key 5 times and verify reading returns the sum |
| `orthogonal_key.py` | Use two orthogonal keys and verify independent reads |
| `correlated_key.py` | Use non orthogonal keys and measure contamination |
| `bench_speed.py` | Recurrent vs parallel timing |
| `bench_memory.py` | GPU memory profiling |

### Gated Linear Attention
| File | Description |
|------|-------------|
| `test_correctness.py` | Matches standard linear attention when alpha=1 |
| `test_full_reset.py` | Setting alpha[r]=0 erases prior state |
| `exponential_decay.py` | State norm decays as alpha^(t) |
| `forgetting.py` | Global gate cannot selectively forget one key |

### Delta Attention
| File | Description |
|------|-------------|
| `test_correctness.py` | Parallel matches recurrent |
| `test_exact_replacement.py` | Same key, new value exactly replaces (beta=1) |
| `test_repeated_consistent.py` | Same key, same value: no change (zero delta) |
| `test_partial_correction.py` | beta=0.25 moves prediction 25% toward target |
| `test_unrelated_preservation.py` | Orthogonal keys: overwriting one doesn't affect the other |
| `test_interference.py` | Correlated keys cause cross-talk |

### Gated DeltaNet
| File | Description |
|------|-------------|
| `test_correctness.py` | Gated DeltaNet matches plain DeltaNet when alphas=ones, betas=ones |
| `test_preserve_overwrite.py` | Overwrite one orthogonal key, the other remains unchanged |
| `test_context_reset.py` | alpha=0 at document boundary erases old context |
| `test_imperfect_reset.py` | Residual recall vs alpha at boundary (sweep 0 to 1) |

### Kimi Delta Attention
| File | Description |
|------|-------------|
| `test_correctness.py` | Recurrent matches chunked |
| `test_chunked.py` | Recurrent vs chunked across sequence lengths |
| `independent_channel_decay.py` | Channels decay independently (S1 = diag(5,2) from diag(10,20)) |
| `exact_overwrite.py` | Same key, new value exactly replaces |
| `one_channel_reset.py` | A single channel can be reset while others persist |
| `anisotropic_forgetting.py` | High alpha forgets slow, low alpha fast |
| `multi_timescale.py` | Half-lives match theory for different alphas |
| `rotated_associations.py` | Rotated keys decay in one dim damages the other |
| `scalar_vs_channel.py` | Per-channel decay beats any single scalar gate |

### Gated DeltaNet-2
| File | Description |
|------|-------------|
| `test_correctness.py` | GDN2 chunked matches Kimi Delta chunked when erase=ones, write=ones |
| `write_only.py` | erase=0, write=1: accumulates without erasing |
| `delete_only.py` | erase=1, write=0: deletes the association |
| `partial_key_erase.py` | Per-channel erase gate removes only selected key channels |
| `partial_value_write.py` | Per-channel write gate commits only selected value channels |
| `recover_gdn.py` | GDN2 reduces to Gated DeltaNet when gates are scalar and equal |

### Comparison Benchmarks (Stage 5)

All models side-by-side using the step-by-step inference interface.

| File | Description |
|------|-------------|
| `exp_A_repeated_overwrite.py` | Same unit key, three overwrites. Additive accumulates, DeltaNet replaces exactly. |
| `exp_B_associative_recall.py` | Store N random pairs, query each. Sweep N and d_k over 10 seeds. Shows how recall degrades when N >> d_k. |
| `exp_C_key_similarity.py` | k1=[1,0], k2=[cosθ, sinθ]. Overwrite k1, measure damage to k2 retrieval vs cosθ. |
| `exp_D_topic_shift.py` | 20 facts for Topic A, then Topic B. Compare ghosting of old context for gated vs non-gated models. |
| `exp_E_runtime.py` | Wall-clock time and stored scalars as sequence grows. Softmax is O(T²), all recurrent models stay O(T). |
| `exp_G2_A_memory_operation_table.py` | INSERT/REPLACE/DELETE/IGNORE via erase gate b and write gate w. GDN2 hits all four; KDA (single beta) only REPLACE. |
| `exp_G2_B_control_surface.py` | Store v_old, apply uncertain v_new. Full (b,w) surface vs KDA's b=w diagonal. |
| `exp_G2_C_selective_commitment.py` | w=[0,1,0,1] commits only location/timestamp. Write gate alone *deletes* untouched fields; masked error (w·(v−prediction)) preserves them. |
| `exp_G2_D_collateral_damage.py` | Delete k_a, measure damage to correlated k_b vs overlap. Full delete: all models identical. Per-channel erase: GDN2 beats uniform beta tradeoff. |

