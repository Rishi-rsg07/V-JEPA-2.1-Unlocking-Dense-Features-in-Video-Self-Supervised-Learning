class LatentPredictor(nn.Module):
    """
    Predicts missing states entirely within the abstract latent space.
    """
    def __init__(self, embed_dim=192):
        super().__init__()
        self.predictor_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=3, dim_feedforward=512, batch_first=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
    def forward(self, context_embeddings, mask):
        # Broadcast mask tokens where patches are hidden
        B, N, D = context_embeddings.shape
        mask_expanded = mask.unsqueeze(-1).float()
        processed_input = (context_embeddings * (1.0 - mask_expanded)) + (self.mask_token * mask_expanded)
        return self.predictor_layer(processed_input)

class DensePredictorLoss(nn.Module):
    """
    Implements the Dense Predictive Loss function. Scores both hidden and visible tokens.
    """
    def __init__(self, lambda_image=0.7, lambda_video=0.5):
        super().__init__()
        self.lambda_image = lambda_image
        self.lambda_video = lambda_video
        
    def forward(self, pred_z, target_z, mask, modality_type):
        abs_diff = torch.abs(pred_z - target_z).mean(dim=-1)
        
        # 1. Standard Masked Loss component
        mask_mask = mask.float()
        loss_predict = (abs_diff * mask_mask).sum() / (mask_mask.sum() + 1e-6)
        
        # 2. Context Loss component (The V-JEPA 2.1 unmasked ground anchor)
        context_mask = (1.0 - mask_mask)
        loss_context_base = (abs_diff * context_mask).sum() / (context_mask.sum() + 1e-6)
        
        lambda_scale = self.lambda_image if modality_type == "image" else self.lambda_video
        loss_context = lambda_scale * loss_context_base
        
        return loss_predict + loss_context, loss_predict.item(), loss_context.item()
        