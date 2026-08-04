import math
import torch
import torch.nn.functional as F


class MemoryModel:
    def reset(self):
        raise NotImplementedError

    def step(self, query, key, value, **controls):
        raise NotImplementedError


class SoftmaxAttention(MemoryModel):
    def __init__(self, d_k, d_v):
        self.d_k = d_k
        self.d_v = d_v
        self.keys = []
        self.values = []
        self.reset()

    def reset(self):
        self.keys = []
        self.values = []

    def step(self, query, key, value, **controls):
        self.keys.append(key)
        self.values.append(value)
        K = torch.stack(self.keys)
        V = torch.stack(self.values)
        scores = query @ K.T / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        return attn @ V


class AdditiveLinear(MemoryModel):
    def __init__(self, d_k, d_v):
        self.d_k = d_k
        self.d_v = d_v
        self.reset()

    def reset(self):
        self.state = torch.zeros(self.d_v, self.d_k)

    def step(self, query, key, value, **controls):
        self.state += torch.outer(value, key)
        return self.state @ query


class GatedAdditive(MemoryModel):
    def __init__(self, d_k, d_v):
        self.d_k = d_k
        self.d_v = d_v
        self.reset()

    def reset(self):
        self.state = torch.zeros(self.d_v, self.d_k)

    def step(self, query, key, value, **controls):
        alpha = controls.get("alpha", 1.0)
        self.state = alpha * self.state + torch.outer(value, key)
        return self.state @ query


class DeltaNet(MemoryModel):
    def __init__(self, d_k, d_v):
        self.d_k = d_k
        self.d_v = d_v
        self.reset()

    def reset(self):
        self.state = torch.zeros(self.d_v, self.d_k)

    def step(self, query, key, value, **controls):
        beta = controls.get("beta", 1.0)
        key = F.normalize(key.unsqueeze(0), 2, 1).squeeze(0)
        prediction = self.state @ key
        self.state += beta * torch.outer(value - prediction, key)
        return self.state @ query


class GatedDeltaNet(MemoryModel):
    def __init__(self, d_k, d_v):
        self.d_k = d_k
        self.d_v = d_v
        self.reset()

    def reset(self):
        self.state = torch.zeros(self.d_v, self.d_k)

    def step(self, query, key, value, **controls):
        alpha = controls.get("alpha", 1.0)
        beta = controls.get("beta", 1.0)
        key = F.normalize(key.unsqueeze(0), 2, 1).squeeze(0)
        prediction = alpha * (self.state @ key)
        self.state = alpha * self.state + beta * torch.outer(value - prediction, key)
        return self.state @ query


class KimiDeltaAttention(MemoryModel):
    def __init__(self, d_k, d_v):
        self.d_k = d_k
        self.d_v = d_v
        self.reset()

    def reset(self):
        self.state = torch.zeros(self.d_k, self.d_v)

    def step(self, query, key, value, **controls):
        alpha = controls.get("alpha", torch.ones(self.d_k))
        beta = controls.get("beta", 1.0)
        if not torch.is_tensor(alpha):
            alpha = torch.full((self.d_k,), alpha)
        key = F.normalize(key.unsqueeze(0), 2, 1).squeeze(0)
        decayed = alpha.unsqueeze(-1) * self.state
        prediction = decayed.T @ key
        error = value - prediction
        self.state = decayed + beta * torch.outer(key, error)
        return self.state.T @ query


class GatedDeltaAttention2(MemoryModel):
    def __init__(self, d_k, d_v):
        self.d_k = d_k
        self.d_v = d_v
        self.reset()

    def reset(self):
        self.state = torch.zeros(self.d_k, self.d_v)

    def step(self, query, key, value, **controls):
        alpha = controls.get("alpha", torch.ones(self.d_k))
        erase_gate = controls.get("erase_gate", torch.ones(self.d_k))
        write_gate = controls.get("write_gate", torch.ones(self.d_v))
        beta = controls.get("beta", 1.0)
        if not torch.is_tensor(alpha):
            alpha = torch.full((self.d_k,), alpha)
        if not torch.is_tensor(erase_gate):
            erase_gate = torch.full((self.d_k,), erase_gate)
        if not torch.is_tensor(write_gate):
            write_gate = torch.full((self.d_v,), write_gate)
        key = F.normalize(key.unsqueeze(0), 2, 1).squeeze(0)
        decayed = alpha.unsqueeze(-1) * self.state
        prediction = decayed.T @ (erase_gate * key)
        error = write_gate * value - prediction
        self.state = decayed + beta * torch.outer(key, error)
        return self.state.T @ query


class SparseDeltaMemory(MemoryModel):
    def __init__(self, memory_size, d_v):
        self.memory_size = memory_size
        self.d_v = d_v
        self.reset()

    def reset(self):
        self.memory = torch.zeros(self.memory_size, self.d_v)

    def step(self, query, key, value, **controls):
        query_indices = controls["query_indices"]  # [R] slots to read
        key_indices = controls["key_indices"]      # [W] slots to write
        alpha = controls.get("alpha", 1.0)
        beta = controls.get("beta", 1.0)
        key = F.normalize(key.unsqueeze(0), 2, 1).squeeze(0)
        decayed = self.memory.clone()
        old_values = decayed[key_indices]
        new_values = alpha * old_values
        prediction = key @ new_values
        error = value - prediction
        decayed[key_indices] = new_values + beta * key.unsqueeze(-1) * error.unsqueeze(0)
        self.memory = decayed
        return (query.unsqueeze(-1) * self.memory[query_indices]).sum(dim=0)
