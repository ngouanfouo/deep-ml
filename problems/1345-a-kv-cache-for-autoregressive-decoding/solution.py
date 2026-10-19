import torch
import torch.nn as nn


class KVCache:
    def __init__(self):
        self.k = None
        self.v = None

    def append(self, k, v):
        if self.k is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def __len__(self):
        return 0 if self.k is None else self.k.shape[2]

    def reset(self):
        self.k = None
        self.v = None


def attend_with_cache(q, k_new, v_new, cache):
    k, v = cache.append(k_new, v_new)
    scale = q.shape[-1] ** 0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)