import torch
import torch.nn as nn


class RNNLayer(nn.Module):
    """Compact GRU wrapper with optional orthogonal initialization."""

    def __init__(self, inputs_dim, outputs_dim, recurrent_N=1, use_orthogonal=True):
        super().__init__()
        self.recurrent_N = recurrent_N
        self.gru = nn.GRU(inputs_dim, outputs_dim, num_layers=recurrent_N)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            else:
                init_fn = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
                init_fn(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        """
        x: (T * N, feat) flattened batch
        hxs: (N, outputs_dim) hidden states
        masks: (T * N, 1) zeros indicate reset points
        """
        N = hxs.size(0)
        T = x.size(0) // N
        x = x.view(T, N, -1)
        masks = masks.view(T, N, 1)

        outputs = []
        hidden = hxs.unsqueeze(0).repeat(self.recurrent_N, 1, 1)
        for t in range(T):
            hidden = hidden * masks[t].transpose(0, 1)
            out, hidden = self.gru(x[t : t + 1], hidden)
            outputs.append(out)

        stacked = torch.cat(outputs, dim=0).view(T * N, -1)
        hidden = hidden[-1]
        return self.norm(stacked), hidden