# Micro-V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning
Micro-V-JEPA 2.1 is a lightweight, self-contained PyTorch implementation of Meta AI's V-JEPA 2.1 paradigm. While traditional Joint Embedding Predictive Architectures (JEPA) excel at capturing high-level global video semantics, they frequently lose fine-grained spatial and temporal details. This repository implements the architectural upgrades introduced in version 2.1 to bridge that gap, enforcing pixel-precise alignment, sharp object boundaries, and temporally consistent tracking.🛠️ Core Architectural Pillars1. Dense Predictive Loss (Section 2.3.1)In previous JEPA versions, the predictive loss was computed exclusively on masked tokens, which allowed visible context tokens to behave as generic global aggregators. V-JEPA 2.1 forces all tokens (both visible context and occluded tokens alike) to contribute to the objective function. By predicting target representations over the entire spatial grid coordinate matrix, the model preserves strict localized physical grounding.$$\mathcal{L}_{\text{dense}} = \mathcal{L}_{\text{masked\_tokens}} + \lambda \cdot \mathcal{L}_{\text{visible\_context\_tokens}}$$2. Deep Self-Supervision (Section 2.3.2)To prevent fine-grained spatial characteristics from being abstracted away in the deeper layers of the network, the self-supervised objective is applied hierarchically across multiple intermediate layers of the encoder (Layers 2, 4, and 6) and fused via a projection MLP.3. Native Multi-Modal Tokenization (Section 2.3.4)Features are extracted natively based on the incoming stream structure without artificial sequence padding. Static images are tokenized using a 2D patch layout [B, N, D], while multi-frame video blocks utilize 3D spatiotemporal tubelet kernels [B, N, D], anchored by learnable modality indicators.
 

### 📊 Empirical Validation: V-JEPA 2 vs 2.1 Dense Tracking
By implementing the *Dense Predictive Loss* term across all context tokens, our joint optimization curve successfully converges to an overall joint loss floor under **0.50** inside 15 training epochs. 

### 🎨 Spatiotemporal PCA Analysis
When passing a novel validation video sequence through the frozen backbone, the model's performance can be proven without downstream fine-tuning. By projecting the high-dimensional latent space down to 3 major components via Principal Component Analysis (PCA) and mapping them to standard RGB channels, moving geometric elements maintain an entirely isolated, uniform color profile across successive temporal blocks.

This visually validates that the latent space has successfully learned to track dense object boundaries across both space and time natively.

* **Masked Structural Loss:** Settles at `~0.65` due to the natural entropy and uncertainty of guessing fully occluded patch areas.
* **Dense Grounding Loss:** Successfully collapses down to `~0.23`, confirming that unmasked tokens act as powerful anchors that enforce explicit spatial and temporal awareness.

### 📜 Citations & References
https://arxiv.org/abs/2603.14482
