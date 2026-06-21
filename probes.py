class AttentiveSegmentationProbe(nn.Module):
    """
    An Attentive Probe that uses cross-attention to map frozen latent patches 
    directly to dense downstream tasks (like pixel-wise object segmentation).
    """
    def __init__(self, embed_dim=192, num_classes=2):
        super().__init__()
        # Cross-attention layer to pool patch representations
        self.query_projection = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=3, batch_first=True)
        
        # Final classification head per patch
        self.segmentation_head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, dense_features):
        # dense_features shape: [B, N, D]
        B, N, D = dense_features.shape
        queries = self.query_projection.expand(B, N, -1)
        
        # Apply localized cross-attention
        attn_out, _ = self.attention(queries, dense_features, dense_features)
        return self.segmentation_head(attn_out) # [B, N, num_classes]

# --- Quick Verification Loop for Probe ---
print("⚙️ Initializing Downstream Attentive Evaluation Probe...")
probe = AttentiveSegmentationProbe(embed_dim=192, num_classes=2).to(device)
probe_optimizer = optim.AdamW(probe.parameters(), lr=1e-3)
probe_criterion = nn.CrossEntropyLoss()

# Freeze the backbone completely to prove representations are robust
tokenizer.eval()
context_encoder.eval()
fusion_mlp.eval()

# Run a single optimization step on a dummy classification mask
for data, modality in train_loader:
    if modality[0] == "video":
        data = data.to(device)
        with torch.no_grad():
            features = fusion_mlp(context_encoder(tokenizer(data, "video")))
        
        # Generate a dummy binary ground-truth segmentation mask (e.g., target object vs background)
        dummy_gt_mask = (features.mean(dim=-1) > 0.0).long() # [B, N]
        
        # Forward pass through probe
        probe_logits = probe(features) # [B, N, 2]
        
        # Compute loss and step
        probe_loss = probe_criterion(probe_logits.transpose(1, 2), dummy_gt_mask)
        probe_optimizer.zero_grad()
        probe_loss.backward()
        probe_optimizer.step()
        
        print(f"✅ Attentive Probe successfully trained for 1 step. Probe Loss: {probe_loss.item():.4f}")
        break