import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    """Enhanced residual block with normalization and dropout"""

    def __init__(self, dim, dropout=0.1, use_layer_norm=True):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(dim)

        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

        # Initialize with smaller weights for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_layer_norm:
            return x + self.layers(self.norm(x))
        return x + self.layers(x)


class SelfAttentionBlock(nn.Module):
    """Self-attention block for capturing long-range dependencies"""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        return x + self.dropout(attn_out)


class FeatureDiscriminator(nn.Module):
    """Discriminator for adversarial training to improve generation quality"""

    def __init__(self, feature_dim, hidden_dim=256, use_spectral_norm=True):
        super().__init__()

        # Spectral normalization helps stabilize GAN training
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else lambda x: x

        self.main = nn.Sequential(
            norm_fn(nn.Linear(feature_dim, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            norm_fn(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            norm_fn(nn.Linear(hidden_dim // 2, hidden_dim // 4)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim // 4, 1)
        )

        # Initialize weights for better gradient flow
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # For multi-token features, average across tokens
        if x.dim() > 2:
            x = x.mean(dim=1)  # [batch_size, tokens, dim] -> [batch_size, dim]

        return self.main(x)


class ImprovedModalGenerator(nn.Module):
    """Enhanced cross-modal generator with advanced architecture and training techniques"""

    def __init__(self, modality_dims, fusion_hidden_dim=512, num_blocks=3):
        """
        Initialize improved cross-modal generator

        Args:
            modality_dims: Dictionary mapping modality names to feature dimensions
                           e.g., {'image': 768, 'text': 512}
            fusion_hidden_dim: Hidden dimension for fusion layers
            num_blocks: Number of residual blocks in encoders and generators
        """
        super().__init__()
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.fusion_hidden_dim = fusion_hidden_dim

        # Create enhanced encoders for each modality
        self.encoders = nn.ModuleDict({
            mod_name: self._build_encoder(dim, fusion_hidden_dim, num_blocks)
            for mod_name, dim in modality_dims.items()
        })

        # Create enhanced generators for each modality pair
        self.generators = nn.ModuleDict()
        for source_mod in self.modalities:
            for target_mod in self.modalities:
                if source_mod != target_mod:
                    self.generators[f"{source_mod}_to_{target_mod}"] = self._build_generator(
                        fusion_hidden_dim,
                        modality_dims[target_mod],
                        num_blocks + (1 if source_mod == 'text' and target_mod == 'image' else 0)
                        # Extra capacity for text→image
                    )

        # Create prior generators for when both modalities are missing
        self.prior_generators = nn.ModuleDict({
            mod_name: self._build_prior_generator(dim, fusion_hidden_dim)
            for mod_name, dim in modality_dims.items()
        })

        # Create discriminators for adversarial training
        self.discriminators = nn.ModuleDict({
            mod_name: FeatureDiscriminator(dim)
            for mod_name, dim in modality_dims.items()
        })

        # Create distribution statistics trackers
        self.register_buffer('image_mean', torch.zeros(modality_dims['image']))
        self.register_buffer('image_std', torch.ones(modality_dims['image']))
        self.register_buffer('text_mean', torch.zeros(modality_dims['text']))
        self.register_buffer('text_std', torch.ones(modality_dims['text']))

        # Reference embeddings for prototype-based generation
        self.register_buffer('image_prototypes', torch.zeros(10, modality_dims['image']))
        self.register_buffer('text_prototypes', torch.zeros(10, modality_dims['text']))

        # Track training statistics
        self.register_buffer('steps', torch.tensor(0))

        # Initialize mixup coefficients for feature interpolation
        self.alpha = 0.2

        # Initialize with smaller weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with smaller values for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _build_encoder(self, input_dim, hidden_dim, num_blocks):
        """Build an enhanced encoder with residual connections and normalization"""
        layers = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        ]

        # Add residual blocks
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim))

        # Add self-attention for tokens if needed
        layers.append(SelfAttentionBlock(hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))

        return nn.Sequential(*layers)

    def _build_generator(self, input_dim, output_dim, num_blocks):
        """Build an enhanced generator with residual connections"""
        layers = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1)
        ]

        # Add residual blocks
        for i in range(num_blocks):
            layers.append(ResidualBlock(input_dim * 2, dropout=0.1))

        # Add self-attention for better coherence
        layers.append(SelfAttentionBlock(input_dim * 2))

        # Output projection
        layers.extend([
            nn.LayerNorm(input_dim * 2),
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        ])

        return nn.Sequential(*layers)

    def _build_prior_generator(self, output_dim, hidden_dim):
        """Build a generator that creates features from noise"""
        return nn.Sequential(
            nn.Linear(256, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(hidden_dim * 2),
            ResidualBlock(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def update_distribution_statistics(self, features):
        """Update running statistics of feature distributions"""
        with torch.no_grad():
            # Update for each modality if features are available
            for mod in self.modalities:
                if mod in features and features[mod] is not None:
                    feat = features[mod]

                    # For multi-token features, average across tokens
                    if feat.dim() > 2:
                        feat = feat.mean(dim=1)

                    # Compute batch statistics
                    batch_mean = feat.mean(dim=0)
                    batch_std = feat.std(dim=0) + 1e-6  # Add epsilon to avoid division by zero

                    # Update running statistics with momentum
                    momentum = min(0.9, (1.0 + self.steps.item()) / (10.0 + self.steps.item()))

                    if mod == 'image':
                        self.image_mean = momentum * self.image_mean + (1 - momentum) * batch_mean
                        self.image_std = momentum * self.image_std + (1 - momentum) * batch_std
                    elif mod == 'text':
                        self.text_mean = momentum * self.text_mean + (1 - momentum) * batch_mean
                        self.text_std = momentum * self.text_std + (1 - momentum) * batch_std

            # Increment step counter
            self.steps += 1

    def update_prototypes(self, features, missing_type):
        """Update prototype features for each modality"""
        with torch.no_grad():
            # Get real features only
            for mod in self.modalities:
                if mod in features and features[mod] is not None:
                    # Skip missing modalities
                    if mod == 'image':
                        real_mask = ~((missing_type == 1) | (missing_type == 3))
                    elif mod == 'text':
                        real_mask = ~((missing_type == 2) | (missing_type == 3))
                    else:
                        continue

                    if not real_mask.any():
                        continue

                    real_features = features[mod][real_mask]

                    # For multi-token features, average across tokens
                    if real_features.dim() > 2:
                        real_features = real_features.mean(dim=1)

                    # Update prototypes with reservoir sampling
                    num_prototypes = 10
                    num_real = real_features.size(0)

                    if num_real == 0:
                        continue

                    for i in range(min(num_real, num_prototypes)):
                        # Simple update using the current batch
                        prototype_tensor = getattr(self, f"{mod}_prototypes")
                        momentum = 0.9
                        prototype_tensor[i] = momentum * prototype_tensor[i] + (1 - momentum) * real_features[
                            i % num_real]

    def normalize_features(self, features, modality):
        """Normalize features using stored statistics"""
        if modality == 'image':
            return (features - self.image_mean) / (self.image_std + 1e-6)
        elif modality == 'text':
            return (features - self.text_mean) / (self.text_std + 1e-6)
        return features

    def denormalize_features(self, features, modality):
        """Denormalize features using stored statistics"""
        if modality == 'image':
            return features * self.image_std + self.image_mean
        elif modality == 'text':
            return features * self.text_std + self.text_mean
        return features

    def encode(self, features, modality):
        """Encode features of a specific modality to the shared space"""
        if features is None:
            return None

        # Handle multi-token features
        if features.dim() > 2:  # [batch_size, tokens, dim]
            batch_size, token_count, feat_dim = features.shape

            # Process each token through the encoder
            features = features.view(-1, feat_dim)  # [batch_size*tokens, dim]
            encoded = self.encoders[modality](features)
            return encoded.view(batch_size, token_count, -1)  # [batch_size, tokens, hidden_dim]
        else:
            return self.encoders[modality](features)

    def generate(self, source_features, source_modality, target_modality, add_noise=True, temperature=1.0):
        """Generate features of target modality from source modality features"""
        # Check if source features are zeros or None
        is_zeros = False
        if source_features is not None:
            is_zeros = torch.sum(torch.abs(source_features)) < 1e-6

        if source_features is None or is_zeros:
            # Use prototype-based generation with noise
            batch_size = 1
            if source_features is not None:
                batch_size = source_features.size(0) if source_features.dim() > 1 else 1

            # Generate from noise with prototype guidance
            noise = torch.randn(batch_size, 256, device=self.image_mean.device)
            raw_features = self.prior_generators[target_modality](noise)

            # Add prototype information for more realistic features
            prototype_idx = torch.randint(0, 10, (batch_size,), device=self.image_mean.device)
            prototypes = getattr(self, f"{target_modality}_prototypes")[prototype_idx]

            # Mix noise-based features with prototype features
            mix_ratio = torch.rand(batch_size, 1, device=self.image_mean.device) * 0.7 + 0.15  # Between 0.15 and 0.85
            if raw_features.dim() > 2 and prototypes.dim() < 3:
                # Expand prototypes to match token dimension
                prototypes = prototypes.unsqueeze(1).expand(-1, raw_features.size(1), -1)
                mix_ratio = mix_ratio.unsqueeze(1)

            mixed_features = mix_ratio * raw_features + (1 - mix_ratio) * prototypes

            # Apply denormalization to match the feature distribution
            return self.denormalize_features(mixed_features, target_modality)

        # Encode source features
        encoded = self.encode(source_features, source_modality)

        # Apply temperature scaling for controlled diversity
        if temperature != 1.0:
            encoded = encoded / temperature

        # Generate target features
        generator = self.generators[f"{source_modality}_to_{target_modality}"]
        generated = generator(encoded)

        # Add controlled noise for diversity and robustness
        if add_noise:
            noise_level = 0.03 * (1.0 - math.exp(-self.steps.item() / 1000))  # Gradually reduce noise
            noise = torch.randn_like(generated) * noise_level
            generated = generated + noise

        # Apply feature statistics to match distribution
        if self.steps.item() > 100:  # Only apply after burn-in period
            generated = self.denormalize_features(generated, target_modality)

        return generated

    def forward(self, features, missing_type):
        """
        Forward pass for feature generation based on missing modalities

        Args:
            features: Dictionary of features for each modality
            missing_type: Integer or tensor indicating missing modality type
                         (0=none, 1=image, 2=text, 3=both)

        Returns:
            Tuple of (generated_features, reconstructed_features, aux_outputs)
            - generated_features: Dictionary of generated features
            - reconstructed_features: Dictionary of reconstructed features
            - aux_outputs: Additional outputs for training (discriminator scores, etc.)
        """
        # Process single sample vs. batch
        batch_mode = isinstance(missing_type, torch.Tensor) and missing_type.dim() > 0

        if not batch_mode:
            # Process single sample
            missing_type = int(missing_type)
            generated, reconstructed, aux_outputs = self._generate_for_sample(features, missing_type)
            return generated, reconstructed, aux_outputs
        else:
            # Process batch
            batch_size = missing_type.size(0)
            device = missing_type.device

            # Update distribution statistics if in training mode
            if self.training:
                # Create copies to avoid modifying input tensors
                feature_copies = {}
                for mod, feat in features.items():
                    if feat is not None:
                        feature_copies[mod] = feat.clone().detach()
                    else:
                        feature_copies[mod] = None

                self.update_distribution_statistics(feature_copies)
                self.update_prototypes(feature_copies, missing_type)

            # Initialize output dictionaries with empty lists
            generated_features = {mod: [] for mod in self.modality_dims.keys()}
            reconstructed_features = {mod: [] for mod in self.modality_dims.keys()}

            # Initialize auxiliary outputs
            aux_outputs = {
                'disc_real_scores': {mod: [] for mod in self.modality_dims.keys()},
                'disc_fake_scores': {mod: [] for mod in self.modality_dims.keys()},
                'distribution_loss': 0.0,
                'cycle_consistency_loss': 0.0
            }

            # Process each sample individually to avoid in-place issues
            for b in range(batch_size):
                # Extract features for this sample
                sample_features = {}
                for mod in features:
                    if features[mod] is not None:
                        # Clone to prevent in-place modifications
                        if features[mod].dim() > 1:
                            sample_features[mod] = features[mod][b:b + 1].clone()
                        else:
                            sample_features[mod] = features[mod].unsqueeze(0).clone()
                    else:
                        sample_features[mod] = None

                # Generate for this sample
                mt = missing_type[b].item()
                gen_sample, recon_sample, sample_aux = self._generate_for_sample(sample_features, mt)

                # Collect generated features
                for mod in gen_sample:
                    if gen_sample[mod] is not None:
                        generated_features[mod].append(gen_sample[mod])

                # Collect reconstructed features
                for mod in recon_sample:
                    if recon_sample[mod] is not None:
                        reconstructed_features[mod].append(recon_sample[mod])

                # Accumulate auxiliary outputs
                for mod in self.modality_dims.keys():
                    if 'disc_real_scores' in sample_aux and mod in sample_aux['disc_real_scores']:
                        aux_outputs['disc_real_scores'][mod].append(sample_aux['disc_real_scores'][mod])
                    if 'disc_fake_scores' in sample_aux and mod in sample_aux['disc_fake_scores']:
                        aux_outputs['disc_fake_scores'][mod].append(sample_aux['disc_fake_scores'][mod])

                # Accumulate scalar losses (divide by batch_size at the end)
                if 'distribution_loss' in sample_aux:
                    aux_outputs['distribution_loss'] += sample_aux['distribution_loss'] / batch_size
                if 'cycle_consistency_loss' in sample_aux:
                    aux_outputs['cycle_consistency_loss'] += sample_aux['cycle_consistency_loss'] / batch_size

            # Convert lists to tensors for each modality
            for mod in self.modality_dims.keys():
                # Process generated features
                if generated_features[mod]:
                    # Check if all tensors have the same shape
                    shapes = [t.shape for t in generated_features[mod]]
                    if all(s == shapes[0] for s in shapes):
                        # If shapes match, concatenate along batch dimension
                        generated_features[mod] = torch.cat(generated_features[mod], dim=0)
                    else:
                        # If shapes don't match, keep as list
                        pass
                else:
                    generated_features[mod] = None

                # Process reconstructed features
                if reconstructed_features[mod]:
                    # Check if all tensors have the same shape
                    shapes = [t.shape for t in reconstructed_features[mod]]
                    if all(s == shapes[0] for s in shapes):
                        # If shapes match, concatenate along batch dimension
                        reconstructed_features[mod] = torch.cat(reconstructed_features[mod], dim=0)
                    else:
                        # If shapes don't match, keep as list
                        pass
                else:
                    reconstructed_features[mod] = None

                # Process discriminator scores
                if aux_outputs['disc_real_scores'][mod]:
                    # Check if all tensors have the same shape
                    shapes = [t.shape for t in aux_outputs['disc_real_scores'][mod]]
                    if all(s == shapes[0] for s in shapes):
                        # If shapes match, concatenate along batch dimension
                        aux_outputs['disc_real_scores'][mod] = torch.cat(aux_outputs['disc_real_scores'][mod], dim=0)
                    else:
                        # If shapes don't match, keep as list
                        pass
                else:
                    aux_outputs['disc_real_scores'][mod] = None

                if aux_outputs['disc_fake_scores'][mod]:
                    # Check if all tensors have the same shape
                    shapes = [t.shape for t in aux_outputs['disc_fake_scores'][mod]]
                    if all(s == shapes[0] for s in shapes):
                        # If shapes match, concatenate along batch dimension
                        aux_outputs['disc_fake_scores'][mod] = torch.cat(aux_outputs['disc_fake_scores'][mod], dim=0)
                    else:
                        # If shapes don't match, keep as list
                        pass
                else:
                    aux_outputs['disc_fake_scores'][mod] = None

            return generated_features, reconstructed_features, aux_outputs

    def _generate_for_sample(self, features, missing_type):
        """Generate features for a single sample based on missing type"""
        token_count = 5  # Number of tokens to use
        device = next(self.parameters()).device

        # Prepare features and handle multi-token case
        processed_features = {}
        for mod in self.modality_dims.keys():
            if mod in features and features[mod] is not None:
                feat = features[mod]

                # Handle different input shapes
                if feat.dim() == 1:  # Single vector
                    processed_features[mod] = feat.unsqueeze(0).clone()  # [1, dim]
                elif feat.dim() == 2:
                    if feat.size(0) == 1:  # Already [1, dim]
                        processed_features[mod] = feat.clone()
                    else:  # [tokens, dim] - take first token_count tokens
                        processed_features[mod] = feat[:token_count].unsqueeze(0).clone()  # [1, tokens, dim]
                elif feat.dim() == 3:  # [batch(1), tokens, dim]
                    # Take first token_count tokens
                    token_len = min(feat.size(1), token_count)
                    processed_features[mod] = feat[:, :token_len].clone()
                else:
                    raise ValueError(f"Unexpected feature shape: {feat.shape}")
            else:
                processed_features[mod] = None

        # Initialize outputs
        generated = {mod: (processed_features[mod].clone() if processed_features[mod] is not None else None)
                     for mod in self.modality_dims.keys()}

        # Additional outputs for training
        aux_outputs = {
            'disc_real_scores': {},
            'disc_fake_scores': {},
            'distribution_loss': 0.0,
            'cycle_consistency_loss': 0.0
        }

        # Generate missing modalities
        if missing_type == 1:  # Image missing, text present
            if processed_features.get("text") is not None:
                text_feat = processed_features["text"]
                generated["image"] = self.generate(text_feat, "text", "image")

                # Run discriminator on generated image if in training mode
                if self.training:
                    # Detach first to avoid backprop through generator for disc loss
                    gen_img = generated["image"].detach()
                    gen_img_flat = gen_img.reshape(-1, gen_img.size(-1))
                    fake_img_score = self.discriminators["image"](gen_img_flat)
                    aux_outputs['disc_fake_scores']["image"] = fake_img_score

        elif missing_type == 2:  # Text missing, image present
            if processed_features.get("image") is not None:
                image_feat = processed_features["image"]
                generated["text"] = self.generate(image_feat, "image", "text")

                # Run discriminator on generated text if in training mode
                if self.training:
                    # Detach first to avoid backprop through generator for disc loss
                    gen_txt = generated["text"].detach()
                    gen_txt_flat = gen_txt.reshape(-1, gen_txt.size(-1))
                    fake_txt_score = self.discriminators["text"](gen_txt_flat)
                    aux_outputs['disc_fake_scores']["text"] = fake_txt_score

        elif missing_type == 3:  # Both missing
            # Generate both modalities from noise
            noise = torch.randn(1, 256, device=device)
            generated["image"] = self.generate(None, None, "image")

            # Use the generated image to create matching text
            generated["text"] = self.generate(generated["image"], "image", "text")

            # Run discriminators if in training mode
            if self.training:
                # Detach to avoid backprop through generator for disc loss
                gen_img = generated["image"].detach()
                gen_txt = generated["text"].detach()

                img_flat = gen_img.reshape(-1, gen_img.size(-1))
                txt_flat = gen_txt.reshape(-1, gen_txt.size(-1))

                fake_img_score = self.discriminators["image"](img_flat)
                fake_txt_score = self.discriminators["text"](txt_flat)

                aux_outputs['disc_fake_scores']["image"] = fake_img_score
                aux_outputs['disc_fake_scores']["text"] = fake_txt_score

        # Run discriminators on real features if available
        if self.training:
            for mod in self.modalities:
                if processed_features.get(mod) is not None:
                    real_feat = processed_features[mod].reshape(-1, processed_features[mod].size(-1))
                    # Detach to ensure we don't update real features
                    real_score = self.discriminators[mod](real_feat.detach())
                    aux_outputs['disc_real_scores'][mod] = real_score

        # Calculate cycle consistency (reconstruction)
        reconstructed = {}
        cycle_loss = 0.0

        if missing_type == 0:  # Both modalities present - can compute cycle consistency
            # Image → Text → Image
            img2txt = self.generate(processed_features["image"], "image", "text", add_noise=False)
            txt2img = self.generate(img2txt, "text", "image", add_noise=False)
            img_cycle_loss = F.mse_loss(txt2img, processed_features["image"])

            # Text → Image → Text
            txt2img = self.generate(processed_features["text"], "text", "image", add_noise=False)
            img2txt = self.generate(txt2img, "image", "text", add_noise=False)
            txt_cycle_loss = F.mse_loss(img2txt, processed_features["text"])

            cycle_loss = img_cycle_loss + txt_cycle_loss
            aux_outputs['cycle_consistency_loss'] = cycle_loss

            reconstructed["image"] = txt2img
            reconstructed["text"] = img2txt

        elif missing_type == 1:  # Image missing - reconstruct text
            if generated["image"] is not None and processed_features["text"] is not None:
                img2txt = self.generate(generated["image"], "image", "text", add_noise=False)
                txt_recon_loss = F.mse_loss(img2txt, processed_features["text"])
                aux_outputs['cycle_consistency_loss'] = txt_recon_loss
                reconstructed["text"] = img2txt

        elif missing_type == 2:  # Text missing - reconstruct image
            if generated["text"] is not None and processed_features["image"] is not None:
                txt2img = self.generate(generated["text"], "text", "image", add_noise=False)
                img_recon_loss = F.mse_loss(txt2img, processed_features["image"])
                aux_outputs['cycle_consistency_loss'] = img_recon_loss
                reconstructed["image"] = txt2img

        # Compute distribution matching loss if training
        if self.training:
            dist_loss = 0.0

            # Check if we have generated image features
            if generated["image"] is not None and processed_features.get("image") is None:
                gen_img = generated["image"].reshape(-1, generated["image"].size(-1))
                # Get statistics - use prototypes as reference
                ref_img = self.image_prototypes.mean(dim=0, keepdim=True)
                img_dist_loss = self._distribution_matching_loss(gen_img, ref_img)
                dist_loss += img_dist_loss

            # Check if we have generated text features
            if generated["text"] is not None and processed_features.get("text") is None:
                gen_txt = generated["text"].reshape(-1, generated["text"].size(-1))
                # Get statistics - use prototypes as reference
                ref_txt = self.text_prototypes.mean(dim=0, keepdim=True)
                txt_dist_loss = self._distribution_matching_loss(gen_txt, ref_txt)
                dist_loss += txt_dist_loss

            aux_outputs['distribution_loss'] = dist_loss

        return generated, reconstructed, aux_outputs

    def _distribution_matching_loss(self, generated, reference):
        """Compute a loss to make generated features match reference distribution"""
        # 1. Mean and variance matching
        gen_mean = generated.mean(dim=0)
        gen_var = generated.var(dim=0)

        ref_mean = reference.mean(dim=0)
        ref_var = reference.var(dim=0) + 1e-6

        mean_loss = F.mse_loss(gen_mean, ref_mean)
        var_loss = F.mse_loss(gen_var, ref_var)

        # 2. Add regularization to prevent degenerate solutions
        diversity_loss = -torch.log(
            torch.det(torch.cov(generated.T) + torch.eye(generated.size(1), device=generated.device) * 1e-6))

        return mean_loss + 0.5 * var_loss + 0.1 * diversity_loss

    def compute_losses(self, real_features, generated_features, aux_outputs, missing_type):
        """
        Compute all losses for generator training

        Args:
            real_features: Dictionary of real features
            generated_features: Dictionary of generated features
            aux_outputs: Auxiliary outputs from forward pass
            missing_type: Tensor indicating missing modality type for each sample

        Returns:
            Dictionary of loss components and total loss
        """
        device = next(self.parameters()).device
        batch_size = missing_type.size(0)

        # Initialize loss components
        losses = {
            'gen_adv_loss': 0.0,
            'cycle_consistency_loss': aux_outputs['cycle_consistency_loss'],
            'distribution_loss': aux_outputs['distribution_loss'],
            'feature_matching_loss': 0.0,
            'total_loss': 0.0
        }

        # 1. Adversarial loss for generator (fool the discriminator)
        gen_adv_loss = 0.0

        for mod in self.modalities:
            if mod in aux_outputs['disc_fake_scores'] and aux_outputs['disc_fake_scores'][mod] is not None:
                # Use hinge loss for GAN stability
                fake_scores = aux_outputs['disc_fake_scores'][mod]
                gen_adv_loss += -fake_scores.mean()

        losses['gen_adv_loss'] = gen_adv_loss

        # 2. Feature matching loss (match real and generated feature distributions)
        feature_matching_loss = 0.0

        # For image missing samples, match text-to-image generated features with real image features
        is_image_missing = (missing_type == 1)
        if is_image_missing.any() and 'image' in generated_features and generated_features['image'] is not None:
            # Find samples with real image features
            has_real_image = ~is_image_missing
            if has_real_image.any() and 'image' in real_features and real_features['image'] is not None:
                # Get real image features
                real_img_feats = real_features['image'][has_real_image]

                # Flatten if multi-token
                if real_img_feats.dim() > 2:
                    real_img_feats = real_img_feats.view(-1, real_img_feats.size(-1))

                # Get generated image features
                gen_img_feats = generated_features['image'][is_image_missing]

                # Flatten if multi-token
                if gen_img_feats.dim() > 2:
                    gen_img_feats = gen_img_feats.view(-1, gen_img_feats.size(-1))

                # Compute feature matching loss
                if real_img_feats.size(0) > 0 and gen_img_feats.size(0) > 0:
                    # Compute mean and covariance for real and generated features
                    real_mean = real_img_feats.mean(dim=0)
                    real_cov = torch.cov(real_img_feats.T) + torch.eye(real_img_feats.size(1), device=device) * 1e-6

                    gen_mean = gen_img_feats.mean(dim=0)
                    gen_cov = torch.cov(gen_img_feats.T) + torch.eye(gen_img_feats.size(1), device=device) * 1e-6

                    # Mean distance
                    mean_dist = F.mse_loss(gen_mean, real_mean)

                    # Frobenius norm of covariance difference
                    cov_dist = torch.norm(real_cov - gen_cov, p='fro') / real_img_feats.size(1)

                    feature_matching_loss += mean_dist + 0.1 * cov_dist

        # Similarly for text missing samples
        is_text_missing = (missing_type == 2)
        if is_text_missing.any() and 'text' in generated_features and generated_features['text'] is not None:
            # Find samples with real text features
            has_real_text = ~is_text_missing
            if has_real_text.any() and 'text' in real_features and real_features['text'] is not None:
                # Get real text features
                real_txt_feats = real_features['text'][has_real_text]

                # Flatten if multi-token
                if real_txt_feats.dim() > 2:
                    real_txt_feats = real_txt_feats.view(-1, real_txt_feats.size(-1))

                # Get generated text features
                gen_txt_feats = generated_features['text'][is_text_missing]

                # Flatten if multi-token
                if gen_txt_feats.dim() > 2:
                    gen_txt_feats = gen_txt_feats.view(-1, gen_txt_feats.size(-1))

                # Compute feature matching loss
                if real_txt_feats.size(0) > 0 and gen_txt_feats.size(0) > 0:
                    # Compute mean and covariance for real and generated features
                    real_mean = real_txt_feats.mean(dim=0)
                    real_cov = torch.cov(real_txt_feats.T) + torch.eye(real_txt_feats.size(1), device=device) * 1e-6

                    gen_mean = gen_txt_feats.mean(dim=0)
                    gen_cov = torch.cov(gen_txt_feats.T) + torch.eye(gen_txt_feats.size(1), device=device) * 1e-6

                    # Mean distance
                    mean_dist = F.mse_loss(gen_mean, real_mean)

                    # Frobenius norm of covariance difference
                    cov_dist = torch.norm(real_cov - gen_cov, p='fro') / real_txt_feats.size(1)

                    feature_matching_loss += mean_dist + 0.1 * cov_dist

        losses['feature_matching_loss'] = feature_matching_loss

        # 3. Combine all losses with weights
        # Weights can be adjusted based on training progress
        cycle_weight = 10.0
        adv_weight = 1.0
        distr_weight = 1.0
        feat_weight = 5.0

        # Decrease cycle consistency weight over time to focus more on adversarial loss
        if self.steps.item() > 1000:
            cycle_weight = max(1.0, 10.0 * math.exp(-self.steps.item() / 5000))

        total_loss = (
                adv_weight * gen_adv_loss +
                cycle_weight * losses['cycle_consistency_loss'] +
                distr_weight * losses['distribution_loss'] +
                feat_weight * feature_matching_loss
        )

        losses['total_loss'] = total_loss
        return losses

    def compute_discriminator_loss(self, aux_outputs):
        """
        Compute discriminator loss

        Args:
            aux_outputs: Auxiliary outputs from forward pass

        Returns:
            Dictionary of discriminator losses
        """
        disc_losses = {}
        total_disc_loss = 0.0

        for mod in self.modalities:
            # Skip modalities without discriminator scores
            if (mod not in aux_outputs['disc_real_scores'] or aux_outputs['disc_real_scores'][mod] is None or
                    mod not in aux_outputs['disc_fake_scores'] or aux_outputs['disc_fake_scores'][mod] is None):
                continue

            # Get scores
            real_scores = aux_outputs['disc_real_scores'][mod]
            fake_scores = aux_outputs['disc_fake_scores'][mod]

            # Compute hinge loss
            real_loss = torch.relu(1.0 - real_scores).mean()
            fake_loss = torch.relu(1.0 + fake_scores).mean()

            disc_loss = real_loss + fake_loss
            disc_losses[f'disc_{mod}_loss'] = disc_loss
            total_disc_loss += disc_loss

        disc_losses['total_disc_loss'] = total_disc_loss
        return disc_losses

    def train_step(self, real_features, missing_type, generator_optimizer, discriminator_optimizer,
                   train_generator=True, train_discriminator=True):
        """
        Perform a complete training step

        Args:
            real_features: Dictionary of real features
            missing_type: Tensor indicating missing modality type for each sample
            generator_optimizer: Optimizer for generator parameters
            discriminator_optimizer: Optimizer for discriminator parameters
            train_generator: Whether to train the generator in this step
            train_discriminator: Whether to train the discriminator in this step

        Returns:
            Dictionary of losses
        """
        # Zero gradients
        if train_generator:
            generator_optimizer.zero_grad()
        if train_discriminator:
            discriminator_optimizer.zero_grad()

        # Forward pass - detach real_features to avoid in-place modifications affecting original tensors
        detached_real_features = {}
        for mod in real_features:
            if real_features[mod] is not None:
                detached_real_features[mod] = real_features[mod].clone()
            else:
                detached_real_features[mod] = None

        # Run forward pass with detached inputs
        generated_features, reconstructed_features, aux_outputs = self(detached_real_features, missing_type)

        # Create separate dictionaries for losses to avoid in-place modifications
        disc_losses = {}
        gen_losses = {}

        # Train discriminator
        if train_discriminator:
            disc_losses = self.compute_discriminator_loss(aux_outputs)
            if 'total_disc_loss' in disc_losses:
                # Use clone to avoid in-place operations
                loss_to_backward = disc_losses['total_disc_loss'].clone()
                loss_to_backward.backward(retain_graph=train_generator)
                discriminator_optimizer.step()

        # Train generator
        if train_generator:
            gen_losses = self.compute_losses(detached_real_features, generated_features, aux_outputs, missing_type)
            if 'total_loss' in gen_losses:
                # Use clone to avoid in-place operations
                loss_to_backward = gen_losses['total_loss'].clone()
                loss_to_backward.backward()
                generator_optimizer.step()

        # Combine all losses
        all_losses = {}
        all_losses.update(disc_losses)
        all_losses.update(gen_losses)

        return all_losses, generated_features, reconstructed_features


# This provides a complete fix for the in-place modification issue
# This is a comprehensive replacement for the CycleGenerationModel class

class CycleGenerationModel(nn.Module):
    """Enhanced cycle generation model with improved architecture and adversarial training"""

    def __init__(self, modality_dims, fusion_hidden_dim=512):
        super().__init__()
        self.modality_dims = modality_dims
        self.generator = ImprovedModalGenerator(modality_dims, fusion_hidden_dim)

        # Track separate parameters for generator and discriminator
        generator_params = []
        discriminator_params = []

        for name, param in self.generator.named_parameters():
            if 'discriminator' in name:
                discriminator_params.append(param)
            else:
                generator_params.append(param)

        self.generator_params = generator_params
        self.discriminator_params = discriminator_params

    def forward(self, features, missing_type):
        """
        Forward pass for feature generation and reconstruction

        Args:
            features: Dictionary of modality features
            missing_type: Missing modality type indicator

        Returns:
            Tuple of (generated_features, reconstructed_features)
        """
        # Disable adversarial components during inference
        self.generator.eval()

        # Create copies of features to avoid in-place modifications
        features_copy = {}
        for mod, feat in features.items():
            if feat is not None:
                features_copy[mod] = feat.clone().detach()
            else:
                features_copy[mod] = None

        # Forward pass without computing adversarial loss
        with torch.no_grad():
            generated, reconstructed, _ = self.generator(features_copy, missing_type)

        # Re-enable training mode if needed
        if self.training:
            self.generator.train()

        return generated, reconstructed

    def train_step(self, features, missing_type, generator_optimizer, discriminator_optimizer,
                   train_generator=True, train_discriminator=True):
        """
        Perform a completely separated training step with adversarial training

        Args:
            features: Dictionary of real features
            missing_type: Tensor indicating missing modality type
            generator_optimizer: Optimizer for generator parameters
            discriminator_optimizer: Optimizer for discriminator parameters
            train_generator: Whether to train generator in this step
            train_discriminator: Whether to train discriminator in this step

        Returns:
            Dictionary of losses, generated features, reconstructed features
        """
        # Record original feature tensors
        original_features = {}
        for mod, feat in features.items():
            if feat is not None:
                original_features[mod] = feat.clone().detach()  # Detach to break gradient flow
            else:
                original_features[mod] = None

        # Dictionary to store all losses
        all_losses = {}

        # ===== DISCRIMINATOR TRAINING =====
        if train_discriminator:
            discriminator_optimizer.zero_grad()

            # Generate features without gradients to train discriminator
            with torch.no_grad():
                # Run generator in eval mode to disable dropout, etc.
                self.generator.eval()
                gen_features, _, _ = self.generator(original_features, missing_type)
                self.generator.train()

            # Compute discriminator loss
            disc_losses = self._compute_discriminator_loss(original_features, gen_features, missing_type)

            # Backward and optimize
            if 'total_disc_loss' in disc_losses and disc_losses['total_disc_loss'] > 0:
                disc_losses['total_disc_loss'].backward()
                discriminator_optimizer.step()

            # Add to all losses
            all_losses.update(disc_losses)

        # ===== GENERATOR TRAINING =====
        generated_features = None
        reconstructed_features = None

        if train_generator:
            generator_optimizer.zero_grad()

            # Run forward pass with gradients
            gen_features, recon_features, aux_outputs = self.generator(original_features, missing_type)

            # Compute generator losses
            gen_losses = self._compute_generator_loss(original_features, gen_features, aux_outputs, missing_type)

            # Backward and optimize
            if 'total_gen_loss' in gen_losses and gen_losses['total_gen_loss'] > 0:
                gen_losses['total_gen_loss'].backward()
                generator_optimizer.step()

            # Save generated features for return
            generated_features = gen_features
            reconstructed_features = recon_features

            # Add to all losses
            all_losses.update(gen_losses)

        # If we didn't train the generator, get features for return value
        if generated_features is None:
            with torch.no_grad():
                generated_features, reconstructed_features, _ = self.generator(original_features, missing_type)

        return all_losses, generated_features, reconstructed_features

    def _compute_discriminator_loss(self, real_features, gen_features, missing_type):
        """
        Compute discriminator loss without affecting generator graph

        Args:
            real_features: Dictionary of real features
            gen_features: Dictionary of generated features
            missing_type: Tensor of missing modality types

        Returns:
            Dictionary of discriminator losses
        """
        batch_size = missing_type.size(0)
        device = missing_type.device
        losses = {}

        # Initialize total loss
        total_disc_loss = 0.0

        # For each modality
        for mod in self.modality_dims:
            # Skip if no generated features for this modality
            if mod not in gen_features or gen_features[mod] is None:
                continue

            # Get the discriminator
            discriminator = self.generator.discriminators[mod]

            # Create masks for real and fake features
            if mod == 'image':
                # Image is missing in types 1 and 3
                fake_mask = (missing_type == 1) | (missing_type == 3)
                real_mask = ~fake_mask
            elif mod == 'text':
                # Text is missing in types 2 and 3
                fake_mask = (missing_type == 2) | (missing_type == 3)
                real_mask = ~fake_mask

            # Skip if no real or fake features
            if not fake_mask.any() and not real_mask.any():
                continue

            # Get fake features and scores
            fake_features = gen_features[mod]

            # Prepare fake features for discriminator
            if fake_features.dim() > 2:  # Handle multi-token case
                fake_features = fake_features.mean(dim=1)  # Average across tokens

            # Get fake scores - detach to avoid affecting generator
            fake_scores = discriminator(fake_features.detach())

            # Initialize real scores
            real_scores = None

            # Get real features and scores if available
            if real_mask.any() and mod in real_features and real_features[mod] is not None:
                real_feats = real_features[mod][real_mask]

                # Handle multi-token case
                if real_feats.dim() > 2:
                    real_feats = real_feats.mean(dim=1)

                # Get real scores
                real_scores = discriminator(real_feats)

            # Compute hinge loss for discriminator
            disc_loss = 0.0

            # Real loss: max(0, 1 - D(x))
            if real_scores is not None and len(real_scores) > 0:
                real_loss = torch.nn.functional.relu(1.0 - real_scores).mean()
                disc_loss += real_loss
                losses[f'disc_{mod}_real_loss'] = real_loss.item()

            # Fake loss: max(0, 1 + D(G(z)))
            if fake_mask.any():
                fake_scores_masked = fake_scores[fake_mask]
                if len(fake_scores_masked) > 0:
                    fake_loss = torch.nn.functional.relu(1.0 + fake_scores_masked).mean()
                    disc_loss += fake_loss
                    losses[f'disc_{mod}_fake_loss'] = fake_loss.item()

            # Add to total loss
            if disc_loss > 0:
                total_disc_loss += disc_loss
                losses[f'disc_{mod}_loss'] = disc_loss.item()

        # Store total loss
        losses['total_disc_loss'] = total_disc_loss

        return losses

    def _compute_generator_loss(self, real_features, gen_features, aux_outputs, missing_type):
        """
        Compute generator loss

        Args:
            real_features: Dictionary of real features
            gen_features: Dictionary of generated features
            aux_outputs: Auxiliary outputs from generator forward pass
            missing_type: Tensor of missing modality types

        Returns:
            Dictionary of generator losses
        """
        batch_size = missing_type.size(0)
        device = missing_type.device
        losses = {}

        # Initialize loss components
        adv_loss = 0.0
        cycle_loss = aux_outputs.get('cycle_consistency_loss', 0.0)
        distribution_loss = aux_outputs.get('distribution_loss', 0.0)
        feature_matching_loss = 0.0

        # 1. Adversarial loss: -D(G(z))
        for mod in self.modality_dims:
            # Skip if no discriminator scores
            if 'disc_fake_scores' not in aux_outputs or mod not in aux_outputs['disc_fake_scores']:
                continue

            # Get fake scores
            fake_scores = aux_outputs['disc_fake_scores'][mod]

            # Skip if no fake scores
            if fake_scores is None or len(fake_scores) == 0:
                continue

            # Calculate adversarial loss: -D(G(z))
            mod_adv_loss = -fake_scores.mean()
            adv_loss += mod_adv_loss
            losses[f'gen_{mod}_adv_loss'] = mod_adv_loss.item()

        # 2. Feature matching loss
        for mod in self.modality_dims:
            # Skip if no generated features for this modality
            if mod not in gen_features or gen_features[mod] is None:
                continue

            # Get missing mask for this modality
            if mod == 'image':
                # Image is missing in types 1 and 3
                missing_mask = (missing_type == 1) | (missing_type == 3)
            elif mod == 'text':
                # Text is missing in types 2 and 3
                missing_mask = (missing_type == 2) | (missing_type == 3)

            # Skip if no missing features
            if not missing_mask.any():
                continue

            # Get real features (excluding missing ones)
            real_mask = ~missing_mask
            if real_mask.any() and mod in real_features and real_features[mod] is not None:
                real_feats = real_features[mod][real_mask]

                # Handle multi-token case
                if real_feats.dim() > 2:
                    real_feats = real_feats.mean(dim=1)

                # Get generated features for this modality
                gen_feats = gen_features[mod][missing_mask]

                # Handle multi-token case
                if gen_feats.dim() > 2:
                    gen_feats = gen_feats.mean(dim=1)

                # Skip if no features
                if len(real_feats) == 0 or len(gen_feats) == 0:
                    continue

                # Compute mean and variance matching
                real_mean = real_feats.mean(dim=0)
                real_var = real_feats.var(dim=0)

                gen_mean = gen_feats.mean(dim=0)
                gen_var = gen_feats.var(dim=0)

                # Compute loss
                mean_loss = F.mse_loss(gen_mean, real_mean)
                var_loss = F.mse_loss(gen_var, real_var)

                # Add to feature matching loss
                mod_feat_loss = mean_loss + 0.5 * var_loss
                feature_matching_loss += mod_feat_loss
                losses[f'gen_{mod}_feat_match_loss'] = mod_feat_loss.item()

        # Store individual loss components
        losses['gen_adv_loss'] = adv_loss.item()
        losses['gen_cycle_loss'] = cycle_loss.item() if isinstance(cycle_loss, torch.Tensor) else cycle_loss
        losses['gen_distribution_loss'] = distribution_loss.item() if isinstance(distribution_loss,
                                                                                 torch.Tensor) else distribution_loss
        losses['gen_feature_matching_loss'] = feature_matching_loss.item()

        # Compute total generator loss with weights
        adv_weight = 1.0
        cycle_weight = 10.0
        dist_weight = 1.0
        feat_weight = 5.0

        total_gen_loss = (
                adv_weight * adv_loss +
                cycle_weight * cycle_loss +
                dist_weight * distribution_loss +
                feat_weight * feature_matching_loss
        )

        losses['total_gen_loss'] = total_gen_loss

        return losses