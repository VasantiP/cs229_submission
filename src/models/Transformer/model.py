import torch
import torch.nn as nn


class ResAttnTransformer(nn.Module):
    def __init__(self, d_emb=512, d_inner=128, d_model=128,
                 n_heads=4, n_layers=2, dim_ff=256,
                 dropout=0.3, max_len=512, score_hidden=32,
                 layer_drop=0.0, causal=False, alibi=False):
        super().__init__()
        self.layer_drop = layer_drop
        self.causal     = causal
        self.alibi      = alibi
        self.n_heads    = n_heads

        self.input_proj = nn.Linear(d_emb, d_inner, bias=False)
        self.res_norm   = nn.LayerNorm(d_inner)
        self.score_net  = nn.Sequential(
            nn.Linear(d_inner, score_hidden), nn.Tanh(), nn.Linear(score_hidden, 1)
        )
        self.value_proj = nn.Sequential(nn.Linear(d_inner, d_model), nn.Dropout(dropout))

        if not alibi:
            self.pos_emb = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.time_score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    @staticmethod
    def _alibi_bias(n_heads, T, device):
        slopes = torch.pow(2.0, -8.0 * torch.arange(1, n_heads + 1, device=device, dtype=torch.float32) / n_heads)
        pos    = torch.arange(T, device=device, dtype=torch.float32)
        dist   = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        return -slopes.view(n_heads, 1, 1) * dist

    def forward(self, esmif, res_mask, time_mask, frame_idxs):
        B, T, R, _ = esmif.shape
        x = esmif.view(B * T, R, -1)
        m = res_mask.view(B * T, R)
        x = self.res_norm(self.input_proj(x))
        scores = self.score_net(x).squeeze(-1).masked_fill(~m, float("-inf"))
        alpha  = torch.softmax(scores, dim=-1)
        frame_emb = (alpha.unsqueeze(-1) * self.value_proj(x)).sum(1)

        if self.alibi:
            h        = frame_emb.view(B, T, -1)
            src_mask = self._alibi_bias(self.n_heads, T, h.device)
        else:
            h        = frame_emb.view(B, T, -1) + self.pos_emb(frame_idxs)
            src_mask = None

        tm = torch.zeros(B, T, device=h.device).masked_fill(time_mask, float("-inf"))

        if self.causal:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
            src_mask    = causal_mask if src_mask is None else causal_mask + src_mask
            for layer in self.layers:
                if self.training and self.layer_drop > 0 and torch.rand(1).item() < self.layer_drop:
                    continue
                h = layer(h, src_mask=src_mask, src_key_padding_mask=tm)
            lengths = (~time_mask).sum(dim=1).clamp(min=1)
            pooled  = h[torch.arange(B, device=h.device), lengths - 1]
        else:
            for layer in self.layers:
                if self.training and self.layer_drop > 0 and torch.rand(1).item() < self.layer_drop:
                    continue
                h = layer(h, src_mask=src_mask, src_key_padding_mask=tm)
            t_scores = self.time_score(h).squeeze(-1).masked_fill(time_mask, float("-inf"))
            beta     = torch.softmax(t_scores, dim=-1)
            pooled   = (beta.unsqueeze(-1) * h).sum(1)

        return self.head(pooled)
