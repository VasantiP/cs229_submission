import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mambapy.mamba import ResidualBlock as _MambaResBlock, MambaConfig as _MambaConfig
    MAMBA_AVAILABLE = True
    MAMBA_BACKEND   = "mambapy"
except ImportError:
    try:
        from mamba_ssm import Mamba as _MambaSSM
        MAMBA_AVAILABLE = True
        MAMBA_BACKEND   = "mamba_ssm"
    except ImportError:
        MAMBA_AVAILABLE = False
        MAMBA_BACKEND   = None


class MeanPooling(nn.Module):
    def __init__(self, d_emb, d_model, dropout):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_emb),
            nn.Linear(d_emb, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        n = mask.sum(-1, keepdim=True).clamp(min=1).float()
        return self.proj(x.sum(1) / n)


class MeanVarPooling(nn.Module):
    def __init__(self, d_emb, d_model, dropout):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(2 * d_emb),
            nn.Linear(2 * d_emb, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        x    = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        n    = mask.sum(-1, keepdim=True).clamp(min=1).float()
        mean = x.sum(1) / n
        var  = ((x - mean.unsqueeze(1)) ** 2 * mask.unsqueeze(-1).float()).sum(1) / n
        return self.proj(torch.cat([mean, var.sqrt().clamp(min=1e-6)], dim=-1))


class AttentionPooling(nn.Module):
    def __init__(self, d_emb, d_inner, d_model, score_hidden, dropout):
        super().__init__()
        self.input_proj = nn.Linear(d_emb, d_inner, bias=False)
        self.res_norm   = nn.LayerNorm(d_inner)
        self.score_net  = nn.Sequential(
            nn.Linear(d_inner, score_hidden), nn.Tanh(),
            nn.Linear(score_hidden, 1),
        )
        self.value_proj = nn.Sequential(nn.Linear(d_inner, d_model), nn.Dropout(dropout))

    def forward(self, x, mask):
        x      = self.res_norm(self.input_proj(x))
        scores = self.score_net(x).squeeze(-1).masked_fill(~mask, float("-inf"))
        alpha  = torch.softmax(scores, dim=-1)
        return (alpha.unsqueeze(-1) * self.value_proj(x)).sum(1)


class ConvPooling(nn.Module):
    def __init__(self, d_emb, d_model, dropout, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(d_emb, d_model, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = F.gelu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        out = out * mask.unsqueeze(-1).float()
        n   = mask.sum(-1, keepdim=True).clamp(min=1).float()
        return self.drop(self.norm(out.sum(1) / n))


class TransformerResPooling(nn.Module):
    def __init__(self, d_emb, d_inner, d_model, score_hidden, dropout,
                 n_res_layers=1, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(d_emb, d_inner, bias=False)
        self.res_norm   = nn.LayerNorm(d_inner)
        res_layer = nn.TransformerEncoderLayer(
            d_model=d_inner, nhead=n_heads, dim_feedforward=d_inner * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.res_transformer = nn.TransformerEncoder(res_layer, num_layers=n_res_layers)
        self.score_net = nn.Sequential(
            nn.Linear(d_inner, score_hidden), nn.Tanh(),
            nn.Linear(score_hidden, 1),
        )
        self.value_proj = nn.Sequential(nn.Linear(d_inner, d_model), nn.Dropout(dropout))

    def forward(self, x, mask):
        x      = self.res_norm(self.input_proj(x))
        x      = self.res_transformer(x, src_key_padding_mask=~mask)
        scores = self.score_net(x).squeeze(-1).masked_fill(~mask, float("-inf"))
        alpha  = torch.softmax(scores, dim=-1)
        return (alpha.unsqueeze(-1) * self.value_proj(x)).sum(1)


class MambaClassifier(nn.Module):
    def __init__(self, pooling, d_emb=512, d_inner=128, d_model=128,
                 n_layers=2, d_state=16, d_conv=4, expand=2,
                 dropout=0.3, max_len=512, score_hidden=32, conv_kernel=5,
                 n_res_layers=1, n_res_heads=4):
        super().__init__()

        if pooling == "mean":
            self.res_pool = MeanPooling(d_emb, d_model, dropout)
        elif pooling == "meanvar":
            self.res_pool = MeanVarPooling(d_emb, d_model, dropout)
        elif pooling == "attention":
            self.res_pool = AttentionPooling(d_emb, d_inner, d_model, score_hidden, dropout)
        elif pooling == "conv":
            self.res_pool = ConvPooling(d_emb, d_model, dropout, conv_kernel)
        elif pooling == "resattn":
            self.res_pool = TransformerResPooling(
                d_emb, d_inner, d_model, score_hidden, dropout,
                n_res_layers=n_res_layers, n_heads=n_res_heads,
            )
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.pos_emb = nn.Embedding(max_len, d_model)

        if MAMBA_BACKEND == "mambapy":
            cfg = _MambaConfig(d_model=d_model, n_layers=1, d_state=d_state,
                               d_conv=d_conv, expand_factor=expand)
            self.mamba_layers = nn.ModuleList([_MambaResBlock(cfg) for _ in range(n_layers)])
            self.mamba_norms  = None
        else:
            self.mamba_layers = nn.ModuleList([
                _MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ])
            self.mamba_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, esmif, res_mask, time_mask, frame_idxs):
        B, T, R, _ = esmif.shape
        frame_emb  = self.res_pool(esmif.view(B * T, R, -1),
                                   res_mask.view(B * T, R)).view(B, T, -1)

        h        = frame_emb + self.pos_emb(frame_idxs)
        pad_mask = (~time_mask).unsqueeze(-1).float()
        if MAMBA_BACKEND == "mambapy":
            for layer in self.mamba_layers:
                h = layer(h) * pad_mask
        else:
            for norm, mamba in zip(self.mamba_norms, self.mamba_layers):
                h = (h + mamba(norm(h))) * pad_mask

        lengths = (~time_mask).sum(dim=1).clamp(min=1)
        pooled  = h[torch.arange(B, device=h.device), lengths - 1]
        return self.head(pooled)
