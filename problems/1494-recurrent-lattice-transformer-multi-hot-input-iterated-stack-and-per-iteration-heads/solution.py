import torch
import torch.nn as nn


class LatticeTransformer(nn.Module):
    def __init__(self, n, V, d, n_heads, n_layers, n_iters, seed=0):
        super().__init__()
        torch.manual_seed(seed)

        self.inp = nn.Linear(V, d)

        self.row_emb = nn.Parameter(torch.randn(n, d) * 0.02)
        self.col_emb = nn.Parameter(torch.randn(n, d) * 0.02)

        self.cls = nn.Parameter(torch.zeros(1, 1, d))

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=n_heads,
                dim_feedforward=4 * d,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d)
        self.cand_head = nn.Linear(d, V)
        self.cls_head = nn.Linear(d, 1)

        self.n = n
        self.V = V
        self.n_iters = n_iters

    def embed(self, x):
        B, n, _, _ = x.shape
        if not x.is_floating_point():
            x = x.float()

        cells = self.inp(x)  # (B, n, n, d)
        cells = cells + self.row_emb.view(1, n, 1, -1) + self.col_emb.view(1, 1, n, -1)
        cells = cells.reshape(B, n * n, -1)  # row-major flatten

        cls = self.cls.expand(B, -1, -1)  # (B, 1, d)
        return torch.cat([cls, cells], dim=1)  # (B, 1 + n*n, d)

    def forward(self, x):
        e = self.embed(x)  # (B, 1 + n*n, d)
        B = e.shape[0]
        h = torch.zeros_like(e)

        cand_iters = []
        cls_iters = []

        for _ in range(self.n_iters):
            h = h + e
            for layer in self.layers:
                h = layer(h)

            hn = self.norm(h)

            cell_tokens = hn[:, 1:]  # (B, n*n, d)
            cand_logits = self.cand_head(cell_tokens).reshape(B, self.n, self.n, self.V)

            cls_logits = self.cls_head(hn[:, 0]).squeeze(-1)  # (B,)

            cand_iters.append(cand_logits)
            cls_iters.append(cls_logits)

        b_iters = torch.stack(cand_iters, dim=0)  # (L, B, n, n, V)
        c_iters = torch.stack(cls_iters, dim=0)   # (L, B)
        return b_iters, c_iters