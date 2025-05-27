import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Tuple, List, Optional, Union, Any


class EncoderBlock(nn.Module):
    """
    Encoder block for cross-modal generation
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    """
    Decoder block for cross-modal generation
    """

    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 2

        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class ModalityEncoder(nn.Module):
    """
    Encodes a modality into a shared latent space
    """

    def __init__(self, modality_dim, latent_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = modality_dim * 2

        self.encoder = EncoderBlock(modality_dim, hidden_dim)
        self.projector = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # Handle 3D tensors (batch_size, token_count, dim)
        if x.dim() == 3:
            batch_size, token_count, dim = x.shape
            # Process each token separately
            hidden = self.encoder(x.view(-1, dim)).view(batch_size, token_count, -1)
            # Average over tokens
            hidden = hidden.mean(dim=1)
        else:
            hidden = self.encoder(x)

        latent = self.projector(hidden)
        return latent


class ModalityDecoder(nn.Module):
    """
    Decodes from latent space back to a modality
    """

    def __init__(self, latent_dim, modality_dim, hidden_dim=None, output_tokens=1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = modality_dim * 2

        self.decoder = DecoderBlock(latent_dim, hidden_dim)
        self.projector = nn.Linear(hidden_dim, modality_dim * output_tokens)
        self.output_tokens = output_tokens
        self.modality_dim = modality_dim

    def forward(self, z):
        hidden = self.decoder(z)
        output = self.projector(hidden)

        # Reshape to handle multiple tokens if needed
        if self.output_tokens > 1:
            output = output.view(output.size(0), self.output_tokens, self.modality_dim)

        return output


class Discriminator(nn.Module):
    """
    Discriminator for adversarial training that determines if a feature is real or generated
    """

    def __init__(self, feature_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feature_dim // 2
            hidden_dim = max(64, hidden_dim)  # Ensure minimum size

        self.layers = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # For 3D input, average over tokens
        if x.dim() == 3:
            x = x.mean(dim=1)

        return self.layers(x)


class CrossModalGenerator(nn.Module):
    """
    Generator for cross-modality feature generation
    """

    def __init__(self, source_dim, target_dim, latent_dim=None, hidden_dim=None, output_tokens=1):
        super().__init__()
        if latent_dim is None:
            latent_dim = min(source_dim, target_dim)

        if hidden_dim is None:
            hidden_dim = max(source_dim, target_dim)

        self.encoder = ModalityEncoder(source_dim, latent_dim, hidden_dim)
        self.decoder = ModalityDecoder(latent_dim, target_dim, hidden_dim, output_tokens)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


class CycleGenerationModel(nn.Module):
    """
    Improved cycle generation model for cross-modal feature generation and reconstruction
    """

    def __init__(
            self,
            modality_dims: Dict[str, int],
            latent_dim: Optional[int] = None,
            fusion_hidden_dim: Optional[int] = None,
            output_tokens: int = 5,
            noise_level: float = 0.1,
            use_gan: bool = True
    ):
        super().__init__()

        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.output_tokens = output_tokens
        self.noise_level = noise_level
        self.use_gan = use_gan

        # Set latent dimension to be the minimum of modality dimensions
        if latent_dim is None:
            latent_dim = min(modality_dims.values())
        self.latent_dim = latent_dim

        # Set hidden dimension for fusion
        if fusion_hidden_dim is None:
            fusion_hidden_dim = max(modality_dims.values()) * 2
        self.fusion_hidden_dim = fusion_hidden_dim

        # Create generators for each modality pair
        self.generators = nn.ModuleDict()
        for source in self.modalities:
            for target in self.modalities:
                if source != target:
                    self.generators[f"{source}2{target}"] = CrossModalGenerator(
                        modality_dims[source],
                        modality_dims[target],
                        latent_dim,
                        fusion_hidden_dim,
                        output_tokens
                    )

        # Create reconstructor modules for cycle consistency
        self.reconstructor = FeatureReconstructor(
            modality_dims,
            fusion_hidden_dim,
            output_tokens
        )

        # Create discriminators for GAN training if enabled
        if use_gan:
            self.discriminators = nn.ModuleDict()
            for modality in self.modalities:
                self.discriminators[modality] = Discriminator(
                    modality_dims[modality],
                    fusion_hidden_dim // 2
                )

        # Training parameters
        self.gan_weight = 0.3
        self.cycle_weight = 1.0
        self.identity_weight = 0.5

        # For tracking statistics
        self.register_buffer('gen_count', torch.zeros(1))
        self.register_buffer('cycle_loss_avg', torch.zeros(1))

    def generate(self, features, source_modality, target_modality, add_noise=True):
        """
        Generate features for target modality from source modality

        Args:
            features: Source modality features
            source_modality: Source modality name (e.g., 'image', 'text')
            target_modality: Target modality name
            add_noise: Whether to add noise to the generated features

        Returns:
            Generated features for target modality
        """
        if features is None:
            return None

        # Get the appropriate generator
        generator = self.generators.get(f"{source_modality}2{target_modality}")
        if generator is None:
            raise ValueError(f"No generator for {source_modality} to {target_modality}")

        # Shape checking and fixing for 2D vs 3D tensors
        if features.dim() == 2 and self.output_tokens > 1:
            # Expand to batch_size x 1 x dim
            features = features.unsqueeze(1)

        # Generate features
        encoded = features
        generated = generator(encoded)

        # Apply random noise if requested
        if add_noise and self.training:
            noise_scale = self.noise_level * torch.rand(1, device=generated.device)
            noise = torch.randn_like(generated) * noise_scale
            # Add noise without in-place operation
            generated = generated + noise

        return generated

    def forward(self, features, missing_type):
        """
        Forward pass for the cycle generation model

        Args:
            features: Dictionary of features for each modality
            missing_type: Type of missing modality (0=none, 1=image, 2=text, 3=both)

        Returns:
            tuple (generated_features, reconstructed_features, auxiliary_outputs)
        """
        # Convert single-value missing_type to batch tensor if needed
        if isinstance(missing_type, int):
            missing_type = torch.full((1,), missing_type, device=next(self.parameters()).device)

        batch_size = missing_type.size(0)
        device = missing_type.device

        # Dictionary to store generated features
        generated_features = {modality: None for modality in self.modalities}

        # Dictionary to store reconstructed features
        reconstructed_features = {modality: None for modality in self.modalities}

        # Auxiliary outputs for loss computation
        aux_outputs = {
            'cycle_losses': {},
            'identity_losses': {},
            'gan_losses': {},
            'missing_types': missing_type.detach()
        }

        # Process each sample in the batch separately to handle different missing types
        for i in range(batch_size):
            # Extract features for this sample
            sample_features = {
                modality: (None if features[modality] is None else
                           features[modality][i:i + 1])
                for modality in self.modalities
            }

            mt = missing_type[i].item()

            # Generate and reconstruct based on missing type
            sample_generated, sample_reconstructed, sample_aux = self._generate_for_sample(
                sample_features, mt
            )

            # Collect results
            for modality in self.modalities:
                if sample_generated[modality] is not None:
                    if generated_features[modality] is None:
                        # Initialize tensor with correct shape
                        feat_shape = sample_generated[modality].shape
                        if len(feat_shape) == 3 and feat_shape[0] == 1:
                            # For 3D tensors [1, tokens, dim]
                            generated_features[modality] = torch.zeros(
                                batch_size, feat_shape[1], feat_shape[2], device=device
                            )
                        else:
                            # For 2D tensors [1, dim]
                            generated_features[modality] = torch.zeros(
                                batch_size, feat_shape[1], device=device
                            )

                    # Copy generated features to the right position
                    if len(sample_generated[modality].shape) == 3:
                        generated_features[modality][i] = sample_generated[modality].squeeze(0)
                    else:
                        generated_features[modality][i] = sample_generated[modality]

                # Same for reconstructed features
                if sample_reconstructed[modality] is not None:
                    if reconstructed_features[modality] is None:
                        # Initialize tensor with correct shape
                        feat_shape = sample_reconstructed[modality].shape
                        if len(feat_shape) == 3 and feat_shape[0] == 1:
                            # For 3D tensors [1, tokens, dim]
                            reconstructed_features[modality] = torch.zeros(
                                batch_size, feat_shape[1], feat_shape[2], device=device
                            )
                        else:
                            # For 2D tensors [1, dim]
                            reconstructed_features[modality] = torch.zeros(
                                batch_size, feat_shape[1], device=device
                            )

                    # Copy reconstructed features to the right position
                    if len(sample_reconstructed[modality].shape) == 3:
                        reconstructed_features[modality][i] = sample_reconstructed[modality].squeeze(0)
                    else:
                        reconstructed_features[modality][i] = sample_reconstructed[modality]

            # Collect auxiliary outputs for loss computation
            for loss_type in ['cycle_losses', 'identity_losses', 'gan_losses']:
                for key, value in sample_aux.get(loss_type, {}).items():
                    if key not in aux_outputs[loss_type]:
                        aux_outputs[loss_type][key] = []
                    aux_outputs[loss_type][key].append(value)

        # Average losses across batch
        for loss_type in ['cycle_losses', 'identity_losses', 'gan_losses']:
            for key in list(aux_outputs[loss_type].keys()):
                if aux_outputs[loss_type][key]:
                    aux_outputs[loss_type][key] = sum(aux_outputs[loss_type][key]) / len(aux_outputs[loss_type][key])
                else:
                    del aux_outputs[loss_type][key]

        # Update statistics
        with torch.no_grad():
            self.gen_count += 1
            if 'cycle_total' in aux_outputs.get('cycle_losses', {}):
                self.cycle_loss_avg = (self.cycle_loss_avg * 0.9 +
                                       aux_outputs['cycle_losses']['cycle_total'] * 0.1)

        return generated_features, reconstructed_features, aux_outputs

    def _generate_for_sample(self, features, missing_type):
        """
        Generate and reconstruct features for a single sample

        Args:
            features: Dictionary of features for one sample
            missing_type: Missing modality type (0=none, 1=image, 2=text, 3=both)

        Returns:
            tuple (generated_features, reconstructed_features, auxiliary_outputs)
        """
        generated = {modality: None for modality in self.modalities}
        reconstructed = {modality: None for modality in self.modalities}

        # Auxiliary outputs for loss computation
        aux_outputs = {
            'cycle_losses': {},
            'identity_losses': {},
            'gan_losses': {}
        }

        # Case: No missing modalities
        if missing_type == 0:
            # For complete samples, just pass through the reconstructor for training
            reconstructed = self.reconstructor(features)

            # Identity preservation loss (optional)
            if self.training:
                identity_losses = {}
                for modality in self.modalities:
                    other_modality = [m for m in self.modalities if m != modality][0]
                    # Generate cross-modal and back
                    cross_gen = self.generate(features[modality], modality, other_modality)
                    back_gen = self.generate(cross_gen, other_modality, modality)

                    # Identity loss
                    if features[modality] is not None and back_gen is not None:
                        identity_loss = F.mse_loss(back_gen, features[modality])
                        identity_losses[f"{modality}_identity"] = identity_loss

                if identity_losses:
                    aux_outputs['identity_losses'] = identity_losses

        # Case: Image missing
        elif missing_type == 1 and 'image' in self.modalities and 'text' in self.modalities:
            # Generate image from text
            generated["image"] = self.generate(features["text"], "text", "image")

            # For cycle consistency
            if self.training:
                # Generate text from generated image
                cycle_text = self.generate(generated["image"], "image", "text")

                # Cycle consistency loss
                if features["text"] is not None and cycle_text is not None:
                    cycle_loss = F.mse_loss(cycle_text, features["text"])
                    aux_outputs['cycle_losses']["text_cycle"] = cycle_loss

                # GAN loss if enabled
                if self.use_gan and generated["image"] is not None:
                    # Train discriminator to distinguish real from generated
                    d_loss = self._compute_gan_loss(
                        self.discriminators["image"],
                        features["image"],  # None for missing modality
                        generated["image"],
                        is_real_batch=False
                    )
                    aux_outputs['gan_losses']["image_d"] = d_loss

            # Reconstruct original text
            reconstructed_input = {"image": generated["image"], "text": features["text"]}
            reconstructed = self.reconstructor(reconstructed_input)

        # Case: Text missing
        elif missing_type == 2 and 'image' in self.modalities and 'text' in self.modalities:
            # Generate text from image
            generated["text"] = self.generate(features["image"], "image", "text")

            # For cycle consistency
            if self.training:
                # Generate image from generated text
                cycle_image = self.generate(generated["text"], "text", "image")

                # Cycle consistency loss
                if features["image"] is not None and cycle_image is not None:
                    cycle_loss = F.mse_loss(cycle_image, features["image"])
                    aux_outputs['cycle_losses']["image_cycle"] = cycle_loss

                # GAN loss if enabled
                if self.use_gan and generated["text"] is not None:
                    # Train discriminator to distinguish real from generated
                    d_loss = self._compute_gan_loss(
                        self.discriminators["text"],
                        features["text"],  # None for missing modality
                        generated["text"],
                        is_real_batch=False
                    )
                    aux_outputs['gan_losses']["text_d"] = d_loss

            # Reconstruct original image
            reconstructed_input = {"image": features["image"], "text": generated["text"]}
            reconstructed = self.reconstructor(reconstructed_input)

        # Case: Both modalities missing (rare edge case)
        elif missing_type == 3:
            # Not much we can do here except generate random features
            # In practice, this should be rare
            for modality in self.modalities:
                # Create random features with appropriate dimensions
                dim = self.modality_dims[modality]
                if self.output_tokens > 1:
                    generated[modality] = torch.randn(1, self.output_tokens, dim, device=next(self.parameters()).device)
                else:
                    generated[modality] = torch.randn(1, dim, device=next(self.parameters()).device)

        # Sum up the cycle losses if any
        if aux_outputs['cycle_losses']:
            aux_outputs['cycle_losses']["cycle_total"] = sum(aux_outputs['cycle_losses'].values())

        # Sum up the identity losses if any
        if aux_outputs['identity_losses']:
            aux_outputs['identity_losses']["identity_total"] = sum(aux_outputs['identity_losses'].values())

        # Sum up the GAN losses if any
        if aux_outputs['gan_losses']:
            aux_outputs['gan_losses']["gan_total"] = sum(aux_outputs['gan_losses'].values())

        return generated, reconstructed, aux_outputs

    def _compute_gan_loss(self, discriminator, real_features, generated_features, is_real_batch=True):
        """
        Compute GAN loss for real or generated features

        Args:
            discriminator: Discriminator module
            real_features: Real features (can be None)
            generated_features: Generated features
            is_real_batch: Whether we're processing real samples (True) or generated (False)

        Returns:
            GAN loss value
        """
        if not self.use_gan:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Skip if either is None
        if real_features is None or generated_features is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Compute loss based on whether we're training on real or generated samples
        if is_real_batch:
            # Real samples should be classified as real (1)
            real_pred = discriminator(real_features)
            real_target = torch.ones_like(real_pred)
            loss = F.binary_cross_entropy_with_logits(real_pred, real_target)
        else:
            # Generated samples should be classified as fake (0)
            fake_pred = discriminator(generated_features)
            fake_target = torch.zeros_like(fake_pred)
            loss = F.binary_cross_entropy_with_logits(fake_pred, fake_target)

        return loss

    def train_step(self, batch_data, optimizer):
        """
        Training step for the generator

        Args:
            batch_data: Dictionary of features and missing types
            optimizer: Optimizer for parameter updates

        Returns:
            Dictionary of losses
        """
        features = batch_data['features']
        missing_type = batch_data['missing_type']

        # Forward pass
        optimizer.zero_grad()
        _, _, aux_outputs = self(features, missing_type)

        # Compute total loss
        total_loss = 0.0
        losses = {}

        # Cycle consistency loss
        if 'cycle_losses' in aux_outputs and 'cycle_total' in aux_outputs['cycle_losses']:
            cycle_loss = aux_outputs['cycle_losses']['cycle_total']
            total_loss += self.cycle_weight * cycle_loss
            losses['cycle'] = cycle_loss.item()

        # Identity preservation loss
        if 'identity_losses' in aux_outputs and 'identity_total' in aux_outputs['identity_losses']:
            identity_loss = aux_outputs['identity_losses']['identity_total']
            total_loss += self.identity_weight * identity_loss
            losses['identity'] = identity_loss.item()

        # GAN loss
        if 'gan_losses' in aux_outputs and 'gan_total' in aux_outputs['gan_losses']:
            gan_loss = aux_outputs['gan_losses']['gan_total']
            total_loss += self.gan_weight * gan_loss
            losses['gan'] = gan_loss.item()

        # Backward pass and optimizer step
        if total_loss > 0:
            total_loss.backward()
            optimizer.step()
            losses['total'] = total_loss.item()

        return losses

    def train_discriminator_step(self, batch_data, optimizer):
        """
        Training step for the discriminator

        Args:
            batch_data: Dictionary of features and missing types
            optimizer: Optimizer for discriminator parameters

        Returns:
            Dictionary of discriminator losses
        """
        if not self.use_gan:
            return {'d_loss': 0.0}

        features = batch_data['features']
        missing_type = batch_data['missing_type']

        # Generate features
        optimizer.zero_grad()
        generated_features, _, _ = self(features, missing_type)

        # Compute discriminator loss
        d_loss = 0.0
        d_losses = {}

        for modality in self.modalities:
            if features[modality] is not None and generated_features[modality] is not None:
                # Detach generated features to avoid training generator
                gen_detached = generated_features[modality].detach()

                # Train on real samples
                real_loss = self._compute_gan_loss(
                    self.discriminators[modality],
                    features[modality],
                    None,
                    is_real_batch=True
                )

                # Train on generated samples
                fake_loss = self._compute_gan_loss(
                    self.discriminators[modality],
                    None,
                    gen_detached,
                    is_real_batch=False
                )

                # Total loss for this modality
                modality_loss = real_loss + fake_loss
                d_loss += modality_loss
                d_losses[f'd_{modality}'] = modality_loss.item()

        # Backward pass and optimizer step
        if d_loss > 0:
            d_loss.backward()
            optimizer.step()
            d_losses['d_total'] = d_loss.item()

        return d_losses


class FeatureReconstructor(nn.Module):
    """
    Reconstructs original features from available modalities
    """

    def __init__(self, modality_dims, fusion_dim=None, output_tokens=1):
        super().__init__()

        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.output_tokens = output_tokens

        # Set fusion dimension
        if fusion_dim is None:
            fusion_dim = sum(modality_dims.values())
        self.fusion_dim = fusion_dim

        # Create encoders for each modality
        self.encoders = nn.ModuleDict({
            modality: EncoderBlock(modality_dims[modality], fusion_dim)
            for modality in self.modalities
        })

        # Create decoders for each modality
        self.decoders = nn.ModuleDict({
            modality: DecoderBlock(fusion_dim, modality_dims[modality] * output_tokens)
            for modality in self.modalities
        })

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, features):
        """
        Reconstruct features for all modalities

        Args:
            features: Dictionary of features for each modality

        Returns:
            Dictionary of reconstructed features
        """
        batch_size = None
        device = next(self.parameters()).device

        # Find batch size from the first non-None feature
        for modality in self.modalities:
            if features[modality] is not None:
                if features[modality].dim() == 3:
                    batch_size = features[modality].size(0)
                else:
                    batch_size = features[modality].size(0)
                break

        if batch_size is None:
            # No features available
            return {modality: None for modality in self.modalities}

        # Process each modality
        encoded_features = {}
        for modality in self.modalities:
            if features[modality] is not None:
                feat = features[modality]

                # Handle 3D tensors (batch_size, token_count, dim)
                if feat.dim() == 3:
                    # For multi-token features, process each token and then average
                    b, t, d = feat.shape
                    feat_flat = feat.reshape(-1, d)
                    encoded = self.encoders[modality](feat_flat).reshape(b, t, -1)
                    encoded = encoded.mean(dim=1)  # Average over tokens
                else:
                    encoded = self.encoders[modality](feat)

                encoded_features[modality] = encoded

        # Fuse available modalities
        if not encoded_features:
            # No features available, return None for all modalities
            return {modality: None for modality in self.modalities}

        # Average available features
        fused = torch.stack(list(encoded_features.values())).mean(dim=0)
        fused = self.fusion(fused)

        # Decode to each modality
        reconstructed = {}
        for modality in self.modalities:
            output = self.decoders[modality](fused)

            # Reshape output if needed
            if self.output_tokens > 1:
                output = output.reshape(-1, self.output_tokens, self.modality_dims[modality])

            reconstructed[modality] = output

        return reconstructed