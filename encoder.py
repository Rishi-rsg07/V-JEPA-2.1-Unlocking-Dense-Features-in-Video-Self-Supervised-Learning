class HierarchicalEncoder(nn.Module):
    """
    Extracts multi-scale representations across intermediate layers
    to fulfill the paper's Deep Self-Supervision criteria.
    """
    def __init__(self, embed_dim=192, num_layers=6, active_save_layers=[2, 4, 6]):
        super().__init__()
        self.active_save_layers = active_save_layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=3, dim_feedforward=512, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        layer_outputs = []
        out = x
        for idx, layer in enumerate(self.layers, start=1):
            out = layer(out)
            if idx in self.active_save_layers:
                layer_outputs.append(self.norm(out))
        return torch.stack(layer_outputs, dim=1) # [B, level_count, sequence_length, embed_dim]

class MultiLevelFusionMLP(nn.Module):
    """
    Flattens and maps multi-tier representations back to standard projection targets.
    """
    def __init__(self, embed_dim=192, level_count=3):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * level_count, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, multi_level_x):
        B, L, N, D = multi_level_x.shape
        x_flat = multi_level_x.transpose(1, 2).reshape(B, N, L * D)
        return self.projection(x_flat)