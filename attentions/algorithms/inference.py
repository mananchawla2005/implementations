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
