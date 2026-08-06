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
    sparse_delta_memory/
    hippocampus_linear_attention/
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

Only writes the residual the difference between the new value and the current state's prediction:

    S[t] = S[t-1] + beta[t] * outer(v[t] - S[t-1] @ k[t], k[t])

- Replaces values for repeated keys instead of accumulating
- Uses associative prefix scan for O(T) parallel computation
- Includes NLMS variant (not generally used, not scale invariant)

### Gated DeltaNet

Combines gating with the delta rule. The state is decayed before computing both the prediction and the update:

    S[t] = alpha[t] * S[t-1] + beta[t] * outer(v[t] - alpha[t] * (S[t-1] @ k[t]), k[t])

The two alphas cancel when writing to an existing key (exact replacement regardless of alpha). The gate only matters when no write happens (pure decay) or when reading with a different key (interference direction).

**Chunked variants** (both DeltaNet and Gated DeltaNet):
Process a block of tokens using a triangular solve within each chunk, then update a checkpoint state between chunks. This gives O(C²) per chunk with O(log C) parallel depth internally, while keeping the cross-chunk recurrence sequential.

### Kimi Delta Attention

Per-channel decay with the delta rule. Unlike Gated DeltaNet's scalar gate, the state is decayed elementwise along the key dimension with a per-channel alpha:

    S[t] = alpha[t] ⊙ S[t-1] + beta[t] * outer(k[t], v[t] - (alpha[t] ⊙ S[t-1])ᵀ k[t])

State layout is [d_k, d_v] (transposed relative to the other models), reads use Sᵀ q and writes use outer(k, error).

**Key property**: independent forgetting per key channel. Because the decay is elementwise, each channel can forget at its own rate , a single channel can be reset (alpha=0) while the rest persist, enabling anisotropic (directional) forgetting rather than a global scalar gate.

### Gated DeltaNet-2 (erase/write gates)

Two independent gates decouple "how much old content to erase" from "how much new content to write":

    decayed   = alpha[t] ⊙ S[t-1]
    e_t       = erase_gate[t] ⊙ k[t]      # per-channel gate on the key (prediction/erase side)
    z_t       = write_gate[t] ⊙ v[t]      # per-channel gate on the value (write side)
    error     = z_t - decayedᵀ e_t
    S[t]      = decayed + outer(k[t], error)

- **erase_gate** (per d_k): gates the key used for prediction, so it controls how much of the old association is subtracted (b in the memory-op table).
- **write_gate** (per d_v): gates the value committed, so it controls how much new content is written (w in the memory-op table).

The two scalar gates span a full 2-D control surface , insert (0,1), replace (1,1), delete (1,0), ignore (0,0) , where models with a single scalar gate (Gated DeltaNet, KDA) are limited to the b=w diagonal.

### Sparse Delta Memory

Scales the state of the gated delta rule by replacing the dense key-value outer product with sparse reads and writes to a large explicit memory of N slots:

    M~[i] = alpha[t] * M[i]                     (selected write slots)
    prediction = sum over W slots of k^(j) * M~[slot_j]
    M[slot_j] = M~[slot_j] + beta[t] * k^(j) * (v[t] - prediction)
    y[t] = sum over R slots of q^(j) * M[query_slot_j]

Unselected slots are untouched. The write budget W and read budget R set the per-step cost as O((W+R) x d_v), independent of the memory size N, while capacity grows with N.

**Product-key addressing**: with two score vectors s1, s2 of size sqrt(N), the outer sum yields N scores per slot and topk picks the W or R slots, exploiting topk(s1 (+) s2) = topk(topk(s1) (+) topk(s2)) so the full score matrix is never materialized.

**Key property**: capacity decouples from update cost. Dense GDN pays d_k rows per step and its capacity ceiling is d_k, while SDM pays W+R rows per step yet holds N slots. When N=d_k and all slots are selected it reduces exactly to Gated DeltaNet.

### Hippocampus Linear Attention (HOLA)

A semiparametric memory: a recurrent per-channel delta state (the parametric part) plus an explicit episodic top-k cache (the non-parametric part), with a gate lambda blending the two:

    state path:
      decayed     = alpha[t] ⊙ S[t-1]          (per-channel decay, KDA-like)
      prediction  = decayedᵀ k[t]
      e[t]        = v[t] - prediction          (the surprise)
      S[t]        = decayed + beta[t] * outer(k[t], e[t])

    cache path:
      score       = beta[t] * ||e[t]||          (surprise = write magnitude)
      keep top-w entries by score, evict the lowest
      read        = softmax( rms(q)ᵀ rms(cache keys) / sqrt(d) + sink ) @ cache values

    output      = S[t]ᵀ q[t] + lambda[t] * cache_read

The state path uses L2-normalized keys, the cache path uses RMS-normalized keys with an attention sink. The state is causal and compressed, the cache stores exact items selected by surprise.

**Key property**: the surprise score ranks how much each write moved the state, so the cache retains exactly the items the state cannot predict , important novel facts, not predictable filler. With lambda=0 it reduces to the per-channel delta rule (KDA), with the state disabled it reduces to explicit causal cache attention. The paper reports cache-path sharpening as a major contributor to performance.

## Inference Interface

Located in `algorithms/inference.py`. A step-by-step interface wrapping each model behind a common API:

```python
class MemoryModel:
    def reset(self): ...
    def step(self, query, key, value, **controls): ...
```

Controls let you pass alpha and beta per step. While the parallel algorithms in `algorithms/` are for training (process all T tokens at once via prefix scan), the step-by-step interface is for autoregressive inference with O(1) memory, O(1) compute per token. `SparseDeltaMemory` additionally takes `query_indices` and `key_indices` controls to pick the R read and W write slots per step. `HippocampusLinearAttention` takes `cache_gate` (lambda) to blend the surprise-scored episodic cache with the state.

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

### Sparse Delta Memory
| File | Description |
|------|-------------|
| `untouched_slots.py` | Writing slots {2,7} leaves every other row bitwise unchanged |
| `exact_replacement.py` | alpha=beta=1, unit weight gives M_t[i] = v_t exactly |
| `fractional_key_weight.py` | k^(i)=0.25 moves the slot partially toward v_t |
| `sparse_read.py` | Reading 2 of 8 slots returns their exact weighted sum |
| `dense_reduction.py` | All slots selected reduces exactly to Gated DeltaNet |
| `product_key_topk.py` | Full outer-sum topk matches the efficient version for N up to 1024 |
| `index_collisions.py` | Duplicate write indices use last-wins scatter, not combined weights |
| `capacity_scaling.py` | Fixed load recall improves with N, proportional load stays flat |
| `addressing_collisions.py` | Collision rate vs associations, N, and W |
| `read_write_budget.py` | Sweep W and R, measure recall, runtime, traffic, effective slots |
| `dense_vs_sparse.py` | Matched-cost dense vs sparse as stored pairs grow |

### Hippocampus Linear Attention
| File | Description |
|------|-------------|
| `surprise_equals_update_norm.py` | Surprise norm equals the recurrent write magnitude (beta\|e\|) |
| `non_unit_key_generalization.py` | Without unit-norm keys, beta\|e\| is wrong by a factor of \|k\| |
| `topk_correctness.py` | Online cache matches scores.topk(8), including ties |
| `online_offline_equivalence.py` | Temporal online top-w equals batch offline top-w |
| `capacity_invariant.py` | Cache stays bounded, fields aligned, causal, unique |
| `recover_gated_delta.py` | lambda=0 collapses to Kimi Delta Attention (channel decay) |
| `pure_cache_mode.py` | State off, lambda=1, w>=T equals explicit causal cache attention |
| `exact_item_preservation.py` | Surprise cache keeps a far important item the state forgets |
| `recency_displacement.py` | Surprise cache resists eviction that a sliding-window cache suffers |
| `duplicate_keys.py` | Identical keys blend values, distinct keys recover exactly |
| `surprise_vs_recency.py` | One fact + many fillers: surprise cache retains, state/recency/full-attn lose it |
| `sharpness_sweep.py` | Cache scale c: sharpening helps while target is argmax, brittle after it flips |
| `state_cache_complementarity.py` | State models a linear map, cache stores exceptions, both fail on pure memorization |
| `false_surprise.py` | Noise with large residuals fools plain surprise, utility-aware score fixes it |
| `redundant_cache_entries.py` | Diversity-aware selection beats plain surprise on near-duplicate keys |
| `admission_vs_eviction.py` | Decoupling admission from eviction beats a single top-w score |
| `cache_budget_allocation.py` | beta\|e\| spends budget on rare facts, never on predictable tokens |

### Comparison Benchmarks (Stage 5)

All models side-by-side using the step-by-step inference interface.

| File | Description |
|------|-------------|
| `exp_A_repeated_overwrite.py` | Same unit key, three overwrites. Additive accumulates, DeltaNet replaces exactly. |
| `exp_B_associative_recall.py` | Store N random pairs, query each. Sweep N and d_k over 10 seeds. Shows how recall degrades when N >> d_k. |
| `exp_C_key_similarity.py` | k1=[1,0], k2=[cosθ, sinθ]. Overwrite k1, measure damage to k2 retrieval vs cosθ. |
| `exp_D_topic_shift.py` | 20 facts for Topic A, then Topic B. Compare ghosting of old context for gated vs non-gated models. |
| `exp_E_runtime.py` | Wall-clock time and stored scalars as sequence grows. Softmax is O(T²), all recurrent models stay O(T). |
| `exp_F_memory_operation_table.py` | INSERT/REPLACE/DELETE/IGNORE via erase gate b and write gate w. GDN2 hits all four, KDA (single beta) only REPLACE. |
| `exp_G_control_surface.py` | Store v_old, apply uncertain v_new. Full (b,w) surface vs KDA's b=w diagonal. |
| `exp_H_selective_commitment.py` | w=[0,1,0,1] commits only location/timestamp. Write gate alone *deletes* untouched fields, masked error (w·(v−prediction)) preserves them. |
| `exp_I_collateral_damage.py` | Delete k_a, measure damage to correlated k_b vs overlap. Full delete: all models identical. Per-channel erase: GDN2 beats uniform beta tradeoff. |
| `exp_J_sdm_vs_hola.py` | SDM (sparse slot table) vs HOLA (surprise cache) on long-range associative recall. Both retain a far important item, HOLA state alone forgets it. |

