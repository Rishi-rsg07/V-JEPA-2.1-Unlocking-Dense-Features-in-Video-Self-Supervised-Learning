# Set evaluation state
tokenizer.eval()
context_encoder.eval()
fusion_mlp.eval()

# 1. Generate a clean video sequence tracking a moving object
eval_dataset = UnifiedMultiModalDataset(total_samples=5, num_frames=6)
eval_video, sample_style = None, ""
for data, style in eval_dataset:
    if style == "video":
        eval_video, sample_style = data, style
        break

with torch.no_grad():
    # Pass video through our model pipeline
    test_input = eval_video.unsqueeze(0).to(device)
    tokens = tokenizer(test_input, "video")
    dense_features = fusion_mlp(context_encoder(tokens)).squeeze(0).cpu().numpy()

# 2. Project high-dimensional features down to 3 components (RGB) using PCA
pca_processor = PCA(n_components=3)
reduced_rgb_features = pca_processor.fit_transform(dense_features)

# Normalize components into clean [0, 1] RGB visualization channels
min_val = reduced_rgb_features.min(axis=0)
max_val = reduced_rgb_features.max(axis=0)
normalized_rgb = (reduced_rgb_features - min_val) / (max_val - min_val + 1e-8)

# --- FIX: Dynamically calculate frames_count based on total sequence length ---
patches_per_dim = 8 
total_patches_per_frame = patches_per_dim * patches_per_dim  # 8 * 8 = 64
total_tokens = normalized_rgb.shape[0]

# Calculate exactly how many frames are actually in this token sequence
frames_count = total_tokens // total_patches_per_frame

# Reshape flattened patches back into a 2D spatial grid map for display
reshaped_visual_grid = normalized_rgb.reshape(frames_count, patches_per_dim, patches_per_dim, 3)

# 3. Plot the PCA tracking maps across the time timeline
fig, axes = plt.subplots(1, frames_count, figsize=(4 * frames_count, 4))
fig.suptitle("🎨 V-JEPA 2.1 Dense Object Tracking Probe (Spatio-Temporal PCA Map)", fontsize=14, fontweight='bold')

# Handle case where there might only be 1 frame output to avoid indexing crashes
if frames_count == 1:
    axes = [axes]

for t in range(frames_count):
    axes[t].imshow(reshaped_visual_grid[t], interpolation='nearest')
    axes[t].set_title(f"Temporal Block {t+1}")
    axes[t].axis('off')

plt.tight_layout()
plt.show()