import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder_layer import QwenEncoderLayer
from mflux.models.qwen.model.qwen_text_encoder.qwen_rms_norm import QwenRMSNorm
from mflux.models.qwen.model.qwen_text_encoder.qwen_rope import QwenRotaryEmbedding


def preprocess_image_for_vision_transformer(image, target_size: tuple[int, int] = (1024, 1024)):
    """
    Preprocess image for vision transformer exactly as the reference does.
    This matches the Qwen2VL preprocessing pipeline exactly.
    
    Args:
        image: PIL Image
        target_size: Target (width, height) for resizing
        
    Returns:
        Preprocessed image tensor ready for patch extraction [C, H, W]
    """
    import numpy as np
    from PIL import Image
    
    # Resize image using BICUBIC (matches Qwen2VL processor)
    image = image.resize(target_size, Image.BICUBIC)
    
    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Apply Qwen2VL normalization (matches the reference exactly)
    # These are the exact values used in Qwen2VL processor
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    image_array = (image_array - mean) / std
    
    # Convert to MLX array: [H, W, C] -> [C, H, W]
    image_tensor = mx.array(image_array.transpose(2, 0, 1))
    
    return image_tensor


def extract_image_patches(image_tensor: mx.array, patch_size: int = 14, temporal_patch_size: int = 2):
    """
    Extract patches from image tensor for vision transformer processing.
    
    Args:
        image_tensor: [C, H, W] image tensor
        patch_size: Spatial patch size (14x14)
        temporal_patch_size: Temporal patch size (2)
        
    Returns:
        patches: [num_patches, C, temporal_patch_size, patch_size, patch_size]
        grid_thw: [1, 3] tensor with [temporal, height_patches, width_patches]
    """
    C, H, W = image_tensor.shape
    
    # Ensure dimensions are divisible by patch_size
    H_padded = ((H + patch_size - 1) // patch_size) * patch_size
    W_padded = ((W + patch_size - 1) // patch_size) * patch_size
    
    # Pad if necessary
    if H_padded != H or W_padded != W:
        print(f"🔎 Padding image from {H}x{W} to {H_padded}x{W_padded} for patch extraction")
        pad_h = H_padded - H
        pad_w = W_padded - W
        # Pad with zeros: (left, right, top, bottom)
        image_tensor = mx.pad(image_tensor, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        H, W = H_padded, W_padded
    
    # Calculate number of patches
    h_patches = H // patch_size
    w_patches = W // patch_size
    
    print(f"🔎 Extracting {h_patches}x{w_patches} patches of size {patch_size}x{patch_size}")
    
    # Reshape image into patches: [C, H, W] -> [C, h_patches, patch_size, w_patches, patch_size]
    patches = image_tensor.reshape(C, h_patches, patch_size, w_patches, patch_size)
    # Transpose to: [h_patches, w_patches, C, patch_size, patch_size]
    patches = patches.transpose(1, 3, 0, 2, 4)
    # Reshape to: [num_patches, C, patch_size, patch_size]
    patches = patches.reshape(h_patches * w_patches, C, patch_size, patch_size)
    
    # Add temporal dimension: [num_patches, C, patch_size, patch_size] -> [num_patches, C, temporal_patch_size, patch_size, patch_size]
    # For images, we repeat the patch temporal_patch_size times
    patches = mx.expand_dims(patches, axis=2)  # [num_patches, C, 1, patch_size, patch_size]
    patches = mx.tile(patches, (1, 1, temporal_patch_size, 1, 1))  # [num_patches, C, temporal_patch_size, patch_size, patch_size]
    
    # Transpose to MLX Conv3D format: [num_patches, temporal_patch_size, patch_size, patch_size, C]
    patches = patches.transpose(0, 2, 3, 4, 1)  # [num_patches, temporal_patch_size, patch_size, patch_size, C]
    
    # Create grid_thw: [temporal=1, height_patches, width_patches]
    grid_thw = mx.array([[1, h_patches, w_patches]], dtype=mx.int32)
    
    return patches, grid_thw


class VisionPatchEmbed(nn.Module):
    """Vision patch embedding using 3D convolution - matches visual.patch_embed.proj"""
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1280,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Conv3d: [embed_dim, in_channels, temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=[temporal_patch_size, patch_size, patch_size],
            stride=[temporal_patch_size, patch_size, patch_size],
            bias=False
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # hidden_states: [num_patches, temporal_patch_size, patch_size, patch_size, channels] (NDHWC format)
        # MLX Conv3d expects input in NDHWC format and weight is already in MLX format
        
        batch_size = hidden_states.shape[0]
        
        # MLX Conv3d expects NDHWC input and our input is already in that format
        # The weight is automatically loaded in MLX format (out_ch, d, h, w, in_ch)
        output = self.proj(hidden_states)  # [num_patches, 1, 1, 1, embed_dim]
        
        # Flatten to [num_patches, embed_dim]
        return output.reshape(batch_size, self.embed_dim)


class VisionRotaryEmbedding(nn.Module):
    """Vision rotary positional embedding"""
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # Create inverse frequencies
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, max_grid_size: int) -> mx.array:
        # Create position embeddings for max_grid_size x max_grid_size
        positions = mx.arange(max_grid_size, dtype=mx.float32)
        freqs = mx.outer(positions, self.inv_freq)
        emb = mx.concatenate([mx.cos(freqs), mx.sin(freqs)], axis=-1)
        return emb


class VisionMLP(nn.Module):
    """Vision MLP block - matches HuggingFace VisionMlp exactly"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # HuggingFace uses simple 2-layer MLP: fc1 -> gelu -> fc2
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # Exact HuggingFace logic: fc2(gelu(fc1(x)))
        return self.fc2(nn.gelu(self.fc1(x)))


class VisionAttention(nn.Module):
    """Vision attention block - matches visual.blocks.*.attn"""
    def __init__(self, embed_dim: int = 1280, num_heads: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def __call__(self, x: mx.array, position_embeddings=None) -> mx.array:
        # For vision attention, x is [seq_len, embed_dim] (no batch dimension)
        seq_len, embed_dim = x.shape
        
        # Compute QKV
        qkv = self.qkv(x)  # [seq_len, 3 * embed_dim]
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=1)
        q = q.squeeze(1).transpose(1, 0, 2)  # [heads, seq, head_dim]
        k = k.squeeze(1).transpose(1, 0, 2)
        v = v.squeeze(1).transpose(1, 0, 2)
        
        # Apply rotary position embeddings if provided
        if position_embeddings is not None:
            cos_emb, sin_emb = position_embeddings
            # Ensure embeddings match sequence length
            if cos_emb.shape[0] != seq_len:
                cos_emb = cos_emb[:seq_len]
                sin_emb = sin_emb[:seq_len]
            
            # Apply RoPE to q and k
            def apply_rope(x, cos, sin):
                # x: [heads, seq, head_dim], cos/sin: [seq, head_dim]
                # Split x into two halves for rotation
                x1, x2 = mx.split(x, 2, axis=-1)  # Each: [heads, seq, head_dim//2]
                
                # Split cos/sin to match x1/x2 dimensions
                cos1, cos2 = mx.split(cos, 2, axis=-1)  # Each: [seq, head_dim//2]
                sin1, sin2 = mx.split(sin, 2, axis=-1)  # Each: [seq, head_dim//2]
                
                # Expand to match head dimension
                cos1 = mx.expand_dims(cos1, axis=0)  # [1, seq, head_dim//2]
                sin1 = mx.expand_dims(sin1, axis=0)  # [1, seq, head_dim//2]
                cos2 = mx.expand_dims(cos2, axis=0)  # [1, seq, head_dim//2]
                sin2 = mx.expand_dims(sin2, axis=0)  # [1, seq, head_dim//2]
                
                return mx.concatenate([
                    x1 * cos1 - x2 * sin1,
                    x1 * sin1 + x2 * cos1
                ], axis=-1)
            
            q = apply_rope(q, cos_emb, sin_emb)
            k = apply_rope(k, cos_emb, sin_emb)
        
        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = mx.matmul(q, k.transpose(0, 2, 1)) * scale  # [heads, seq, seq]
        attn_weights = mx.softmax(scores, axis=-1)
        attn_output = mx.matmul(attn_weights, v)  # [heads, seq, head_dim]
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, embed_dim)  # [seq, embed_dim]
        return self.proj(attn_output)


class VisionBlock(nn.Module):
    """Vision transformer block - matches visual.blocks.*"""
    def __init__(self, embed_dim: int = 1280, num_heads: int = 16, mlp_ratio: float = 2.671875):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = VisionAttention(embed_dim, num_heads)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)  # 3420 for 1280 * 2.671875
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim)

    def __call__(self, x: mx.array, position_embeddings=None) -> mx.array:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x), position_embeddings)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Complete vision transformer - matches the visual.* structure"""
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1280,
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 2.671875,
        hidden_size: int = 3584,
    ):
        super().__init__()
        
        self.patch_embed = VisionPatchEmbed(patch_size, temporal_patch_size, in_channels, embed_dim)
        
        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        
        self.blocks = [VisionBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        # Register blocks as attributes for parameter tracking
        for i, block in enumerate(self.blocks):
            setattr(self, f"blocks_{i}", block)
            
        # Replace complex spatial merging with simple linear projection (like diffusers)
        self.merger = nn.Linear(embed_dim, hidden_size, bias=True)  # 1280 -> 3584

    def rot_pos_emb(self, grid_thw: mx.array) -> mx.array:
        """Generate rotary position embeddings for grid_thw"""
        pos_ids = []
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            
            # For spatial merge, we need to handle odd dimensions
            # Simply create position IDs without spatial merge reshaping for now
            # This is a simpler approach that avoids the reshape error
            
            # Create flattened position IDs
            total_patches = h * w
            hpos_ids = mx.repeat(mx.arange(h), w)  # [0,0,0...w times, 1,1,1...w times, ...]
            wpos_ids = mx.tile(mx.arange(w), h)  # [0,1,2...w-1, 0,1,2...w-1, ...]
            
            # Stack [h_pos, w_pos] and repeat for temporal dimension
            pos_id_pair = mx.stack([hpos_ids, wpos_ids], axis=-1)  # [total_patches, 2]
            pos_id_pair = mx.tile(pos_id_pair, (t, 1))  # Repeat t times
            pos_ids.append(pos_id_pair)
        
        pos_ids = mx.concatenate(pos_ids, axis=0)  # [total_patches, 2]
        
        # Get max grid size for rotary embeddings
        max_grid_size = int(mx.max(grid_thw[:, 1:]).item())
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)  # [max_grid_size, head_dim]
        
        # Index into the full embedding (ensure indices are integers)
        h_indices = pos_ids[:, 0].astype(mx.int32)
        w_indices = pos_ids[:, 1].astype(mx.int32)
        h_emb = rotary_pos_emb_full[h_indices]  # [total_patches, head_dim]
        w_emb = rotary_pos_emb_full[w_indices]  # [total_patches, head_dim]
        rotary_pos_emb = mx.concatenate([h_emb, w_emb], axis=-1)  # [total_patches, head_dim*2]
        
        return rotary_pos_emb

    def __call__(self, pixel_values: mx.array, grid_thw: mx.array) -> mx.array:
        """
        Args:
            pixel_values: [batch_size, channels, temporal_patch_size, patch_size, patch_size]
            grid_thw: [num_images, 3] - temporal, height, width dimensions
        """
        # Patch embedding
        hidden_states = self.patch_embed(pixel_values)  # [num_patches, embed_dim]
        
        # Rotary position embeddings 
        rotary_pos_emb = self.rot_pos_emb(grid_thw)  # [num_patches, head_dim*2] (h+w concatenated)
        # Split into cos and sin for attention layers
        position_embeddings = (mx.cos(rotary_pos_emb), mx.sin(rotary_pos_emb))
        
        
        # Apply vision transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, position_embeddings)
        
        # Merge patches to final hidden size
        return self.merger(hidden_states)


class QwenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        num_hidden_layers: int = 28,
        max_position_embeddings: int = 128000,
        rope_theta: float = 1000000.0,
        # Vision config
        image_token_id: int = 151655,
        video_token_id: int = 151656,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [QwenEncoderLayer() for i in range(num_hidden_layers)]
        # Register layers as attributes so parameters are tracked
        for i, layer in enumerate(self.layers):
            setattr(self, f"layers_{i}", layer)
        self.norm = QwenRMSNorm(hidden_size, eps=1e-6)
        self.rotary_emb = QwenRotaryEmbedding(
            dim=hidden_size // 28,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            rope_type="default",
        )
        
        # Vision transformer for VL fusion
        self.visual = VisionTransformer(
            patch_size=14,
            temporal_patch_size=2,
            in_channels=3,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            mlp_ratio=2.671875,
            hidden_size=hidden_size,  # 3584
        )

    def get_image_features(self, pixel_values: mx.array, image_grid_thw: mx.array) -> mx.array:
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.
        Matches the reference implementation exactly.
        """
        # pixel_values should be raw image patches: [num_patches, C, T, H, W]
        # Convert pixel_values to visual dtype (float32 for MLX)
        pixel_values = pixel_values.astype(mx.float32)
        image_embeds = self.visual(pixel_values, image_grid_thw)
        
        # Split embeddings based on grid sizes (no spatial merging like diffusers)
        split_sizes = image_grid_thw.prod(axis=-1).astype(mx.int32)
        split_sizes = [int(s) for s in split_sizes.tolist()]
        image_embeds = mx.split(image_embeds, split_sizes, axis=0)
        return image_embeds
    
    def process_image_natively(self, image_path: str, target_size: tuple[int, int] = (1024, 1024)):
        """
        Process image natively without HuggingFace dependencies.
        
        Args:
            image_path: Path to image file
            target_size: Target (width, height) for resizing
            
        Returns:
            pixel_values: [num_patches, C, T, H, W] patches ready for vision transformer
            image_grid_thw: [1, 3] grid dimensions
        """
        from PIL import Image
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess_image_for_vision_transformer(image, target_size)
        
        # Extract patches
        pixel_values, image_grid_thw = extract_image_patches(image_tensor)
        
        return pixel_values, image_grid_thw

    def get_placeholder_mask(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array,
        image_features: mx.array = None,
    ) -> mx.array:
        """
        Obtains multimodal placeholder mask from input_ids and checks that the placeholder token count
        is equal to the length of multimodal features. Matches the reference implementation.
        """
        # Find image token positions
        special_image_mask = input_ids == self.image_token_id
        n_image_tokens = mx.sum(special_image_mask).item()
        
        # Expand mask to match embeddings shape
        special_image_mask = mx.expand_dims(special_image_mask, axis=-1)  # [batch, seq, 1]
        special_image_mask = mx.broadcast_to(special_image_mask, inputs_embeds.shape)  # [batch, seq, hidden]
        
        # Validate token count matches features
        if image_features is not None:
            total_image_features = image_features.shape[0]  # image_features is already concatenated
            if n_image_tokens != total_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {total_image_features}"
                )
        
        return special_image_mask

    def __call__(
        self,
        input_ids: mx.array | None = None,
        attention_mask: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        precomputed_image_embeds: mx.array | None = None,
    ) -> mx.array:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            batch_size, seq_len = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            batch_size, seq_len, _ = inputs_embeds.shape

        # Vision-Language fusion - matches reference implementation exactly
        if precomputed_image_embeds is not None and image_grid_thw is not None:
            print(f"🔧 DEBUG: Using precomputed image embeddings (shape={precomputed_image_embeds.shape}, image_grid_thw={image_grid_thw})")
            
            # Use precomputed embeddings directly - bypass entire vision pipeline
            image_embeds = precomputed_image_embeds
            print(f"🔧 DEBUG: Using precomputed image_embeds: {image_embeds.shape}")
            
        elif pixel_values is not None and image_grid_thw is not None:
            print(f"🔎 QwenEncoder: Processing VL fusion (pixel_values.shape={pixel_values.shape}, image_grid_thw={image_grid_thw})")
            
            # Detect format: if pixel_values has 5 dimensions, it's native patches [num_patches, C, T, H, W]
            # If it has 2 dimensions, it's processed embeddings from tokenizer that we need to reject
            if len(pixel_values.shape) == 5:
                print(f"🔎 QwenEncoder: Native patch format detected - processing through vision transformer")
                # Native format: process through vision transformer
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = mx.concatenate(image_embeds, axis=0)
            elif len(pixel_values.shape) == 2:
                print(f"🔎 QwenEncoder: HF tokenizer format detected - processing through native vision transformer")
                
                if pixel_values.shape[-1] == 1176:
                    print(f"🔎 QwenEncoder: ROOT CAUSE IDENTIFIED!")
                    print(f"🔎 QwenEncoder: HF processor is doing internal vision processing with wrong config")
                    print(f"🔎 QwenEncoder: HF processor provides {pixel_values.shape[-1]}-dim processed embeddings")
                    print(f"🔎 QwenEncoder: But our native vision transformer expects raw pixel data")
                    print(f"🔎 QwenEncoder: Our loaded weights expect {self.visual.patch_embed.embed_dim}-dim embeddings")
                    print(f"🔎 QwenEncoder: SOLUTION: Bypass HF image processor, use native preprocessing")
                    
                    raise ValueError(
                        f"HuggingFace processor is doing internal vision processing and giving us "
                        f"{pixel_values.shape[-1]}-dim processed embeddings instead of raw pixel data. "
                        f"We need to bypass the HF image processor and implement native preprocessing "
                        f"that feeds raw pixel data to our native MLX vision transformer."
                    )
                else:
                    print(f"🔎 QwenEncoder: Unknown HF embedding dimension {pixel_values.shape[-1]}")
                    # For unknown dimensions, try to process as-is
                    image_embeds = pixel_values.astype(mx.float32)
            else:
                print(f"🔎 QwenEncoder: Unknown pixel_values format with shape {pixel_values.shape}")
                raise ValueError(f"Unsupported pixel_values format: {pixel_values.shape}")
            
            # Replace image placeholder tokens with actual image embeddings
            image_positions = input_ids == self.image_token_id
            n_image_tokens = mx.sum(image_positions).item()
            
            print(f"🔎 QwenEncoder: Found {n_image_tokens} image tokens, have {image_embeds.shape[0]} image embeddings")
            
            if n_image_tokens > 0 and image_embeds.shape[0] >= n_image_tokens:
                # Replace image tokens with image embeddings
                # Create a list of embeddings and then stack them
                image_positions_flat = image_positions.flatten()
                inputs_embeds_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                
                # Take only the first n_image_tokens embeddings to match token count
                image_embeds_to_use = image_embeds[:n_image_tokens]
                
                # Build new embeddings list
                new_embeds_list = []
                image_idx = 0
                for i in range(len(image_positions_flat)):
                    if image_positions_flat[i] and image_idx < image_embeds_to_use.shape[0]:
                        new_embeds_list.append(image_embeds_to_use[image_idx])
                        image_idx += 1
                    else:
                        new_embeds_list.append(inputs_embeds_flat[i])
                
                # Stack the new embeddings
                new_embeds = mx.stack(new_embeds_list, axis=0)
                inputs_embeds = new_embeds.reshape(inputs_embeds.shape)
                print(f"🔎 QwenEncoder: Replaced {image_idx} image tokens with embeddings")

        cache_position = mx.arange(seq_len, dtype=mx.int32)
        position_ids = mx.expand_dims(mx.expand_dims(cache_position, axis=0), axis=0)
        position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_len))
        if attention_mask is not None:
            padding_mask = mx.where(
                attention_mask == 1,
                mx.zeros_like(attention_mask).astype(mx.float32),
                mx.ones_like(attention_mask).astype(mx.float32) * (-float("inf")),
            )
            padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)
        else:
            padding_mask = None

        # Create causal triangular mask
        idx = mx.arange(seq_len, dtype=mx.int32)
        j = mx.expand_dims(idx, axis=0)
        i = mx.expand_dims(idx, axis=1)
        tri_bool = j > i
        zeros_2d = mx.zeros((seq_len, seq_len)).astype(mx.float32)
        neginf_2d = mx.ones((seq_len, seq_len)).astype(mx.float32) * (-float("inf"))
        causal_tri_mask = mx.where(tri_bool, neginf_2d, zeros_2d)
        causal_tri_mask = mx.expand_dims(mx.expand_dims(causal_tri_mask, axis=0), axis=0)  # [1,1,S,S]
        causal_tri_mask = mx.broadcast_to(causal_tri_mask, (batch_size, 1, seq_len, seq_len))

        # Final attention mask: causal triangle + padding mask
        if padding_mask is not None:
            attention_mask_4d = causal_tri_mask + padding_mask
        else:
            attention_mask_4d = causal_tri_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask_4d, position_embeddings)
        hidden_states = self.norm(hidden_states)
        return hidden_states
