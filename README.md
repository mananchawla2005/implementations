# Implementations

A practical exploration of different architectures and research directions across pretraining and postraining of LLMs.

## Project Structure

```
attentions/
  algorithms/         # Core implementations
  experiments/        # Experiments and tests for each algo to understand it.
    shared/           # Plotting and benchmarking utilities
    linear_attention/
    softmax_attention/
    gated_linear_attention/
    delta_attention/
```

## Algorithms

### Linear Attention

Standard linear attention with state = sum of outer products.

- **Recurrent**: O(T) sequential, O(1) memory per step
- **Parallel**: O(T) via cumulative sum, fully parallelizable

**Key property**: State grows unbounded (no forgetting).

### Gated Linear Attention

Adds a scalar gate alpha[t] per timestep that decays the state before writing:

    S[t] = alpha[t] * S[t-1] + outer(v[t], k[t])

Convention: **decay, then write, then read**.

Uses parallel cumulative-product formulation for O(T) computation.

### Delta Attention

Only writes the residual i.e the difference between the new value and the current state's prediction:

    S[t] = S[t-1] + beta[t] * outer(v[t] - S[t-1] @ k[t], k[t])

- Replaces values for repeated keys instead of accumulating
- Uses associative prefix scan for O(T) parallel computation
- Includes NLMS variant ( but is not generally used as it is not scale invariant )

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
| `test_correctness.py` | Parallel implementation matches recurrent |
| `test_exact_replacement.py` | Same key but new value exactly replaces when (beta=1) |
| `test_repeated_consistent.py` | Same key and same value results in no change (zero delta) |
| `test_partial_correction.py` | beta=0.25 makes prediction move 25% toward target |
| `test_unrelated_preservation.py` | Overwriting one of the orthogonal keys doesn't affect the other |
| `test_interference.py` | Correlated keys causes interference |

