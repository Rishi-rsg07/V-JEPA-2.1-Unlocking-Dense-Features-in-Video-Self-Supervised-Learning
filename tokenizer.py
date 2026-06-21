class MultiModalTokenizer(nn.Module):
    """
    Tokenizes multi-modal input (image or video) into a sequence of patches,
    then linearly projects them to a consistent embedding dimension.
    """
    def __init__(self, embed_dim=192, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # A 3-channel input (RGB), patch_size x patch_size, flattened
        # projects to embed_dim
        self.projection = nn.Linear(3 * patch_size * patch_size, embed_dim)

    def forward(self, x, modality_type):
        # x can be either [B, C, H, W] for image or [B, C, F, H, W] for video
        B = x.shape[0]
        if modality_type == "video":
            # For video, treat frames as additional batch dimension or flatten across frames
            # [B, C, F, H, W] -> [B*F, C, H, W]
            _, C, F, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            
            # Extract patches: [B*F, C, H, W] -> [B*F, num_patches, patch_dim]
            # This creates patches for each frame, then we'll combine them
            patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            patches = patches.contiguous().view(B * F, C, -1, self.patch_size, self.patch_size)
            patches = patches.permute(0, 2, 1, 3, 4).reshape(B * F, -1, C * self.patch_size * self.patch_size)
            
            # Project patches to embedding dimension
            tokens = self.projection(patches)
            
            # Reshape back to B x (F * num_patches_per_frame) x embed_dim
            num_patches_per_frame = tokens.shape[1]
            tokens = tokens.reshape(B, F * num_patches_per_frame, self.embed_dim)

        else: # modality_type == "image"
            # Extract patches: [B, C, H, W] -> [B, num_patches, patch_dim]
            patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            patches = patches.contiguous().view(B, 3, -1, self.patch_size, self.patch_size)
            patches = patches.permute(0, 2, 1, 3, 4).reshape(B, -1, 3 * self.patch_size * self.patch_size)
            
            # Project patches to embedding dimension
            tokens = self.projection(patches)
            
        return tokens

# 1. Instantiate Network Components
embed_dimension = 192
tokenizer = MultiModalTokenizer(embed_dim=embed_dimension).to(device)
context_encoder = HierarchicalEncoder(embed_dim=embed_dimension).to(device)
fusion_mlp = MultiLevelFusionMLP(embed_dim=embed_dimension).to(device)
predictor = LatentPredictor(embed_dim=embed_dimension).to(device)

# Build Target networks with independent parameter spaces
target_encoder = HierarchicalEncoder(embed_dim=embed_dimension).to(device)
target_fusion_mlp = MultiLevelFusionMLP(embed_dim=embed_dimension).to(device)

# Strict gradient isolation for target tracking paths
for p in list(target_encoder.parameters()) + list(target_fusion_mlp.parameters()):
    p.requires_grad = False

# 2. Optimization Settings
optimizer = optim.AdamW(
    list(tokenizer.parameters()) + list(context_encoder.parameters()) +
    list(fusion_mlp.parameters()) + list(predictor.parameters()),
    lr=1e-3, weight_decay=1e-4
)
criterion = DensePredictorLoss()

# 3. Data Pipeline Configuration (Batch size=1 to handle alternating tensor shapes cleanly)
train_dataset = UnifiedMultiModalDataset(total_samples=300)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

print("🏃 Starting Micro-V-JEPA 2.1 Joint Pre-Training Loop...")
for epoch in range(15):
    epoch_total_loss, epoch_pred_loss, epoch_ctx_loss = 0.0, 0.0, 0.0

    tokenizer.train()
    context_encoder.train()
    fusion_mlp.train()
    predictor.train()

    for raw_data, modality_type in train_loader:
        raw_data = raw_data.to(device)
        modality = modality_type[0] # Unwrap batch string tuple

        optimizer.zero_grad()

        # Online context pipeline path
        context_tokens = tokenizer(raw_data, modality)
        online_hierarchy = context_encoder(context_tokens)
        online_z = fusion_mlp(online_hierarchy)

        # Setup target tracking path with zero tracking gradients
        with torch.no_grad():
            target_tokens = tokenizer(raw_data, modality)
            target_hierarchy = target_encoder(target_tokens)
            target_z = target_fusion_mlp(target_hierarchy)

        # Generate random 60% token masking array map
        num_patches = context_tokens.shape[1]
        mask = (torch.rand(1, num_patches, device=device) > 0.4)

        # Forward pass through latent space predictor
        predicted_z = predictor(online_z, mask)

        # Calculate Dense Predictive Loss
        loss, l_pred, l_ctx = criterion(predicted_z, target_z, mask, modality)
        loss.backward()
        optimizer.step()

        # Target Network Exponential Moving Average (EMA) weight tracking step
        with torch.no_grad():
            for c_param, t_param in zip(context_encoder.parameters(), target_encoder.parameters()):
                t_param.data.copy_(0.99 * t_param.data + 0.01 * c_param.data)
            for c_param, t_param in zip(fusion_mlp.parameters(), target_fusion_mlp.parameters()):
                t_param.data.copy_(0.99 * t_param.data + 0.01 * c_param.data)

        epoch_total_loss += loss.item()
        epoch_pred_loss += l_pred
        epoch_ctx_loss += l_ctx

    avg_total = epoch_total_loss / len(train_loader)
    print(f"Epoch {epoch+1:02d}/15 | Combined Loss: {avg_total:.5f} | Masked Component: {epoch_pred_loss/len(train_loader):.5f} | Dense Ground Component: {epoch_ctx_loss/len(train_loader):.5f}")