class UnifiedMultiModalDataset(Dataset):
    """
    Generates synthetic geometric shapes to train multi-modal systems.
    Randomly alternate streams between 4D image structures and 5D video paths.
    """
    def __init__(self, total_samples=200, num_frames=6, canvas_size=64):
        self.total_samples = total_samples
        self.num_frames = num_frames
        self.canvas_size = canvas_size

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 50% probability layout choice between image mode and video sequence
        is_video = np.random.rand() > 0.5
        
        if is_video:
            # 5D Video Pipeline: [Channels, Frames, Height, Width]
            video = torch.zeros(3, self.num_frames, self.canvas_size, self.canvas_size)
            start_x = np.random.randint(10, self.canvas_size - 25)
            start_y = np.random.randint(10, self.canvas_size - 25)
            dx, dy = np.random.choice([-2, -1, 1, 2]), np.random.choice([-2, -1, 1, 2])
            
            # Draw a shifting block across time dimensions
            for t in range(self.num_frames):
                cx = int(start_x + t * dx)
                cy = int(start_y + t * dy)
                video[0, t, cy:cy+16, cx:cx+16] = 0.8  # Red channel trail
                video[1, t, cy+4:cy+12, cx+4:cx+12] = 0.5  # Green overlay channel
                video[2, t, :, :] = 0.1  # Ambient structural noise floor
            return video, "video"
        else:
            # 4D Static Image Pipeline: [Channels, Height, Width]
            image = torch.zeros(3, self.canvas_size, self.canvas_size)
            cx = np.random.randint(10, self.canvas_size - 25)
            cy = np.random.randint(10, self.canvas_size - 25)
            image[0, cy:cy+20, cx:cx+20] = 0.9  # Fixed primary element
            image[2, :, :] = 0.15              # Unique static ambient layer
            return image, "image"

# Verification validation checkout
dataset_preview = UnifiedMultiModalDataset(total_samples=10)
sample_tensor, sample_tag = dataset_preview[0]
print(f"📊 Dataset Sample initialized successfully!")
print(f"   🔹 Output Category Tag: {sample_tag.upper()}")
print(f"   🔹 Tensor Dimension Matrix: {sample_tensor.shape}")