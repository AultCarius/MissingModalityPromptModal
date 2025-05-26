import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention for feature processing"""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attention_mask=None):
        # Self-attention block - avoid inplace operations
        attn_output, _ = self.self_attn(
            query=x, key=x, value=x,
            key_padding_mask=attention_mask if attention_mask is not None else None
        )
        # Use addition instead of inplace operations
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward block
        ffn_output = self.ffn(x)
        x = x + ffn_output  # Not using inplace addition
        x = self.norm2(x)

        return x


# Fix for CrossAttentionLayer to avoid inplace operations
class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for conditioning on the other modality"""

    def __init__(self, query_dim, key_dim, num_heads=8, dropout=0.1):
        super().__init__()
        # Use projection layers if dimensions don't match
        self.need_projection = query_dim != key_dim
        if self.need_projection:
            self.query_proj = nn.Linear(query_dim, key_dim)
            self.out_proj = nn.Linear(key_dim, query_dim)
            self.attn_dim = key_dim
        else:
            self.attn_dim = query_dim

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key_value, query_mask=None, kv_mask=None):
        # Make a copy of the input to avoid modifying it
        query_copy = query.clone()

        # Project query if needed
        attn_query = self.query_proj(query_copy) if self.need_projection else query_copy

        # Cross-attention block
        attn_output, _ = self.cross_attn(
            query=attn_query,
            key=key_value,
            value=key_value,
            key_padding_mask=kv_mask if kv_mask is not None else None,
            attn_mask=query_mask if query_mask is not None else None
        )

        # Project back if needed
        if self.need_projection:
            attn_output = self.out_proj(attn_output)

        # Residual connection and norm - avoid inplace operations
        query_copy = query_copy + self.dropout(attn_output)
        query_copy = self.norm1(query_copy)

        # Feed-forward block
        ffn_output = self.ffn(query_copy)
        query_copy = query_copy + ffn_output  # Not using inplace addition
        query_copy = self.norm2(query_copy)

        return query_copy


class EnhancedCrossModalGenerator(nn.Module):
    """Enhanced cross-modal generator using transformer architecture"""

    def __init__(self, modality_dims, fusion_hidden_dim=512, num_layers=3, num_heads=8):
        """
        Initialize the enhanced cross-modal generator

        Args:
            modality_dims: Dictionary mapping modality names to their dimensions
            fusion_hidden_dim: Dimension of the shared hidden space
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.fusion_dim = fusion_hidden_dim

        # Modality encoders - project to common dimension
        self.encoders = nn.ModuleDict({
            mod_name: nn.Sequential(
                nn.Linear(dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for mod_name, dim in modality_dims.items()
        })

        # Shared transformer layers for initial encoding
        self.shared_encoders = nn.ModuleDict({
            mod_name: nn.ModuleList([
                TransformerEncoderLayer(fusion_hidden_dim, num_heads)
                for _ in range(num_layers // 2)  # Half the layers for encoding
            ]) for mod_name in self.modalities
        })

        # Cross-modal generation transformers
        self.generators = nn.ModuleDict()
        for source_mod in self.modalities:
            for target_mod in self.modalities:
                if source_mod != target_mod:
                    # Create cross-attention based generator
                    layers = []
                    # First add cross-attention to condition on source modality
                    layers.append(CrossAttentionLayer(
                        fusion_hidden_dim, fusion_hidden_dim, num_heads
                    ))
                    # Then add transformer layers to process the conditioned features
                    for _ in range(num_layers - 1):
                        layers.append(TransformerEncoderLayer(
                            fusion_hidden_dim, num_heads
                        ))
                    # Final projection to target modality dimension
                    layers.append(nn.Sequential(
                        nn.LayerNorm(fusion_hidden_dim),
                        nn.Linear(fusion_hidden_dim, modality_dims[target_mod]),
                    ))

                    self.generators[f"{source_mod}_to_{target_mod}"] = nn.ModuleList(layers)

        # Prior generators for "both" missing case - more sophisticated with multiple layers
        self.prior_generators = nn.ModuleDict()
        for mod_name, dim in modality_dims.items():
            layers = []
            # Initial projection from noise
            layers.append(nn.Sequential(
                nn.Linear(256, fusion_hidden_dim),  # Larger noise dimension
                nn.LayerNorm(fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ))
            # Add transformer layers for processing
            for _ in range(num_layers):
                layers.append(TransformerEncoderLayer(
                    fusion_hidden_dim, num_heads
                ))
            # Final projection to target modality dimension
            layers.append(nn.Sequential(
                nn.LayerNorm(fusion_hidden_dim),
                nn.Linear(fusion_hidden_dim, dim),
            ))

            self.prior_generators[mod_name] = nn.ModuleList(layers)

        # Special token for generation (like CLS token)
        self.gen_tokens = nn.ParameterDict({
            mod_name: nn.Parameter(torch.randn(1, 1, fusion_hidden_dim))
            for mod_name in self.modalities
        })

    def encode(self, features, modality, attention_mask=None):
        """
        Encode features of a specific modality to the common space

        Args:
            features: Tensor of shape [batch_size, token_count, dim] or [batch_size, dim]
            modality: Modality name (e.g., 'image' or 'text')
            attention_mask: Optional mask for tokens

        Returns:
            Encoded features in the common space
        """
        if features is None:
            return None

        # Make a clone to avoid modifying the input
        features_clone = features.clone()

        # Handle single vector input
        if features_clone.dim() == 2:
            features_clone = features_clone.unsqueeze(1)  # [batch_size, 1, dim]

        # Project to common dimension
        encoded = self.encoders[modality](features_clone)  # [batch_size, token_count, fusion_dim]

        # Apply transformer layers
        for layer in self.shared_encoders[modality]:
            encoded = layer(encoded, attention_mask)

        return encoded

    def generate(self, source_features, source_modality, target_modality,
                 source_mask=None, inference=False):
        """
        Generate features for the target modality from source modality features

        Args:
            source_features: Source modality features [batch_size, token_count, dim]
            source_modality: Source modality name
            target_modality: Target modality name
            source_mask: Optional attention mask for source tokens
            inference: Whether in inference mode (affects output handling)

        Returns:
            Generated features for target modality
        """
        # Check if source features are available
        is_missing = False
        if source_features is None:
            is_missing = True
        elif torch.sum(torch.abs(source_features)) < 1e-6:
            is_missing = True

        batch_size = 1
        device = next(self.parameters()).device

        if is_missing:
            # Use prior generator for missing source
            noise = torch.randn(batch_size, 5, 256, device=device)  # Multiple noise tokens

            # Process through prior generator
            prior_gen = self.prior_generators[target_modality]
            features = prior_gen[0](noise)  # Initial projection

            # Apply transformer layers
            for i in range(1, len(prior_gen) - 1):
                features = prior_gen[i](features)

            # Final projection to target dimension
            generated = prior_gen[-1](features)

            return generated

        # Get batch size from source features and make a clone to avoid modifying input
        source_features = source_features.clone().detach()

        if source_features.dim() == 3:
            batch_size = source_features.size(0)
        else:
            batch_size = 1
            source_features = source_features.unsqueeze(0)  # Add batch dimension

        # Encode source features
        encoded_source = self.encode(source_features, source_modality, source_mask)

        # Get generator layers
        generator = self.generators[f"{source_modality}_to_{target_modality}"]

        # Add generation token
        gen_token = self.gen_tokens[target_modality].expand(batch_size, -1, -1)
        query = gen_token.clone()  # Clone to avoid inplace modifications

        # First layer is cross-attention
        query = generator[0](query, encoded_source)

        # Apply remaining transformer layers
        for i in range(1, len(generator) - 1):
            query = generator[i](query)

        # Final projection to target dimension
        generated = generator[-1](query)

        # For inference, return in expected format
        if inference and generated.size(1) == 1:
            return generated.squeeze(1)  # Remove token dimension if single token

        return generated

    def forward(self, features, missing_type=None):
        """
        Forward pass for generating missing modalities

        Args:
            features: Dictionary of available features
            missing_type: Missing modality type (0=none, 1=image, 2=text, 3=both)

        Returns:
            Dictionary of generated features
        """
        # Determine batch size and device
        batch_size = 1
        device = next(self.parameters()).device

        # Try to get batch size from features
        for mod, feat in features.items():
            if feat is not None:
                if feat.dim() >= 2:
                    batch_size = feat.size(0)
                    device = feat.device
                    break

        # Create output containers
        generated_features = {mod: None for mod in self.modalities}

        # If batch processing (tensor of missing types)
        if isinstance(missing_type, torch.Tensor) and missing_type.dim() > 0:
            # Initialize output tensors
            for mod in self.modalities:
                if mod in features and features[mod] is not None:
                    token_count = features[mod].size(1) if features[mod].dim() > 2 else 1
                    generated_features[mod] = torch.zeros(
                        batch_size, token_count, self.modality_dims[mod],
                        device=device
                    )

            # Process each sample based on its missing type
            for b in range(batch_size):
                mt = missing_type[b].item()

                # Extract sample features
                sample_features = {
                    mod: features[mod][b:b + 1] if mod in features and features[mod] is not None else None
                    for mod in self.modalities
                }

                # Generate based on missing type
                if mt == 1:  # Image missing
                    if 'text' in sample_features and sample_features['text'] is not None:
                        img_feat = self.generate(
                            sample_features['text'], 'text', 'image', inference=True
                        )
                        generated_features['image'][b] = img_feat

                elif mt == 2:  # Text missing
                    if 'image' in sample_features and sample_features['image'] is not None:
                        txt_feat = self.generate(
                            sample_features['image'], 'image', 'text', inference=True
                        )
                        generated_features['text'][b] = txt_feat

                elif mt == 3:  # Both missing
                    # Generate both modalities from priors
                    for mod in self.modalities:
                        gen_feat = self.generate(None, None, mod, inference=True)
                        generated_features[mod][b] = gen_feat

                # For mt == 0 (none missing), keep original features

        else:
            # Single sample processing
            mt = int(missing_type) if missing_type is not None else 0

            if mt == 1:  # Image missing
                if 'text' in features and features['text'] is not None:
                    generated_features['image'] = self.generate(
                        features['text'], 'text', 'image'
                    )

            elif mt == 2:  # Text missing
                if 'image' in features and features['image'] is not None:
                    generated_features['text'] = self.generate(
                        features['image'], 'image', 'text'
                    )

            elif mt == 3:  # Both missing
                # Generate both modalities from priors
                for mod in self.modalities:
                    generated_features[mod] = self.generate(None, None, mod)

        return generated_features


class TransformerReconstructor(nn.Module):
    """
    Enhanced reconstructor using transformer architecture for cycle consistency
    """

    def __init__(self, modality_dims, fusion_dim=512, num_layers=2, num_heads=8):
        """
        Initialize the transformer-based reconstructor

        Args:
            modality_dims: Dictionary mapping modality names to their dimensions
            fusion_dim: Dimension of fusion space
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())

        # Encoders (shared with generator or separate)
        self.encoders = nn.ModuleDict({
            mod_name: nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for mod_name, dim in modality_dims.items()
        })

        # Reconstruction transformers for each modality
        self.decoders = nn.ModuleDict()
        for source_mod in self.modalities:
            for target_mod in self.modalities:
                if source_mod != target_mod:
                    # Cross-modal reconstruction path
                    layers = []

                    # First add cross-attention layer
                    layers.append(CrossAttentionLayer(
                        fusion_dim, fusion_dim, num_heads
                    ))

                    # Add transformer layers
                    for _ in range(num_layers):
                        layers.append(TransformerEncoderLayer(
                            fusion_dim, num_heads
                        ))

                    # Final projection to target modality
                    layers.append(nn.Sequential(
                        nn.LayerNorm(fusion_dim),
                        nn.Linear(fusion_dim, modality_dims[target_mod]),
                    ))

                    self.decoders[f"{source_mod}_to_{target_mod}"] = nn.ModuleList(layers)

        # Self-reconstruction paths (for complete modality samples)
        for mod_name in self.modalities:
            layers = []

            # Add transformer layers
            for _ in range(num_layers):
                layers.append(TransformerEncoderLayer(
                    fusion_dim, num_heads
                ))

            # Final projection
            layers.append(nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, modality_dims[mod_name]),
            ))

            self.decoders[f"{mod_name}_to_{mod_name}"] = nn.ModuleList(layers)

    def reconstruct(self, features, source_modality, target_modality, source_mask=None):
        """
        Reconstruct target modality features from source modality

        Args:
            features: Source modality features
            source_modality: Source modality name
            target_modality: Target modality name
            source_mask: Optional attention mask for source tokens

        Returns:
            Reconstructed features for target modality
        """
        if features is None:
            return None

        # Make a clone to avoid modifying the input tensor
        features_clone = features.clone()

        # Handle single vector input
        if features_clone.dim() == 2:
            features_clone = features_clone.unsqueeze(1)  # [batch_size, 1, dim]

        # Encode features to fusion space - detach to avoid inplace operations affecting original
        encoded = self.encoders[source_modality](features_clone)

        # Get decoder path
        decoder = self.decoders[f"{source_modality}_to_{target_modality}"]

        # For cross-modal reconstruction
        if source_modality != target_modality:
            # First layer is cross-attention (using self-attention)
            query = encoded.clone()  # Clone to avoid inplace modifications
            query = decoder[0](query, encoded, source_mask)

            # Apply remaining transformer layers
            for i in range(1, len(decoder) - 1):
                query = decoder[i](query, source_mask)

            # Final projection
            reconstructed = decoder[-1](query)
        else:
            # For self-reconstruction
            query = encoded.clone()  # Clone to avoid inplace modifications

            # Apply transformer layers
            for i in range(len(decoder) - 1):
                query = decoder[i](query, source_mask)

            # Final projection
            reconstructed = decoder[-1](query)

        return reconstructed

    def forward(self, features):
        """
        Forward pass to reconstruct all modalities

        Args:
            features: Dictionary of features for different modalities

        Returns:
            Dictionary of reconstructed features
        """
        if not isinstance(features, dict):
            raise ValueError("Expected dictionary of features")

        reconstructed = {}

        # Make a defensive copy of the features dictionary to avoid modifying the original
        features_copy = {}
        for key, value in features.items():
            if value is not None:
                features_copy[key] = value.clone().detach()
            else:
                features_copy[key] = None

        # For each available modality, reconstruct all others
        for source_mod, source_feat in features_copy.items():
            if source_feat is None or torch.sum(torch.abs(source_feat)) < 1e-6:
                continue

            for target_mod in self.modalities:
                # Skip if target is the same as source
                if target_mod == source_mod:
                    continue

                # Skip if already reconstructed
                key = f"{target_mod}_from_{source_mod}"
                if key in reconstructed:
                    continue

                # Reconstruct target from source
                recon_feat = self.reconstruct(source_feat, source_mod, target_mod)

                # Store reconstructed features
                if recon_feat is not None:
                    reconstructed[key] = recon_feat

        return reconstructed


class EnhancedCycleGenerationModel(nn.Module):
    """
    Enhanced cycle generation model with better batch handling and training capabilities
    """

    def __init__(self, modality_dims, fusion_hidden_dim=512, num_layers=3, num_heads=8):
        """
        Initialize the enhanced cycle generation model

        Args:
            modality_dims: Dictionary mapping modality names to their dimensions
            fusion_hidden_dim: Dimension of fusion space
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())

        # Enhanced generator and reconstructor
        self.generator = EnhancedCrossModalGenerator(
            modality_dims, fusion_hidden_dim, num_layers, num_heads
        )

        self.reconstructor = TransformerReconstructor(
            modality_dims, fusion_hidden_dim, num_layers // 2, num_heads
        )

    def forward(self, features, missing_type, generate_all=False):
        """
        Forward pass for generating missing modalities and reconstruction

        Args:
            features: Dictionary of available features
            missing_type: Missing modality type (0=none, 1=image, 2=text, 3=both)
            generate_all: Whether to generate all modalities regardless of missing status

        Returns:
            Tuple of (generated_features, reconstructed_features, cycle_features)
            - generated_features: Generated features for missing modalities
            - reconstructed_features: Reconstructed original features
            - cycle_features: Features after a complete cycle (for cycle consistency)
        """
        # Phase 1: Generate missing modalities - make a defensive copy
        features_copy = {}
        for key, value in features.items():
            if value is not None:
                features_copy[key] = value.clone().detach()
            else:
                features_copy[key] = None

        generated_features = self.generator(features_copy, missing_type)

        # Combine original and generated features without modifying originals
        combined_features = {}
        for mod in self.modalities:
            if mod in features and features[mod] is not None:
                combined_features[mod] = features[mod].clone().detach()
            elif mod in generated_features and generated_features[mod] is not None:
                combined_features[mod] = generated_features[mod].clone().detach()

        # Phase 2: Reconstruct for cycle consistency
        reconstructed_features = self.reconstructor(combined_features)

        # Phase 3 (optional): Generate all modalities for training
        cycle_features = None
        if generate_all:
            all_generated = {}

            # Make another copy to avoid modifying the originals
            features_for_cycle = {}
            for key, value in features.items():
                if value is not None:
                    features_for_cycle[key] = value.clone().detach()
                else:
                    features_for_cycle[key] = None

            # For each modality, generate all others
            for source_mod, source_feat in features_for_cycle.items():
                if source_feat is None:
                    continue

                for target_mod in self.modalities:
                    if source_mod == target_mod:
                        continue

                    # Generate target from source
                    gen_feat = self.generator.generate(
                        source_feat.clone().detach(),  # Make sure we don't modify the original
                        source_mod,
                        target_mod
                    )

                    if gen_feat is not None:
                        key = f"{target_mod}_from_{source_mod}"
                        all_generated[key] = gen_feat

            cycle_features = all_generated

        return generated_features, reconstructed_features, cycle_features

    def generate_for_sample(self, features, missing_type):
        """
        Generate missing modalities for a single sample

        Args:
            features: Dictionary of features for one sample
            missing_type: Missing modality type

        Returns:
            Tuple of (generated, reconstructed)
        """
        # Convert missing_type to int if needed
        mt = int(missing_type) if not isinstance(missing_type, int) else missing_type

        # Generate missing modalities
        generated = {}

        if mt == 1:  # Image missing
            if 'text' in features and features['text'] is not None:
                generated['image'] = self.generator.generate(
                    features['text'], 'text', 'image'
                )

        elif mt == 2:  # Text missing
            if 'image' in features and features['image'] is not None:
                generated['text'] = self.generator.generate(
                    features['image'], 'image', 'text'
                )

        elif mt == 3:  # Both missing
            # Generate both modalities from priors
            for mod in self.modalities:
                generated[mod] = self.generator.generate(None, None, mod)

        # Combine original and generated
        combined = {}
        for mod in self.modalities:
            if mod in features and features[mod] is not None:
                combined[mod] = features[mod]
            elif mod in generated and generated[mod] is not None:
                combined[mod] = generated[mod]

        # Reconstruct for cycle consistency
        reconstructed = self.reconstructor(combined)

        return generated, reconstructed
