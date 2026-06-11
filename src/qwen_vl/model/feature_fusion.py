"""
Feature fusion modules for combining 2D visual features with 3D geometry features.

OVERVIEW
--------
JanusVLN fuses VGGT 3D geometry features into the Qwen2.5-VL vision+language pipeline.
This file provides the fusion modules used by the "deepstack" fusion modes, which inject
geometry features at INTERMEDIATE layers of the vision tower and/or language model
decoder, rather than only at the end (as the original `lam`-additive mode does).

FUSION MODES
------------
There are four deepstack modes, each a string value for `config.feature_fusion_method`:

  1. "deepstack_vision_add"         — inject inside vision transformer blocks (additive)
  2. "deepstack_vision_cross_attn"  — inject inside vision blocks (cross-attention)
  3. "deepstack_language_add"       — inject inside LM decoder blocks (additive)
  4. "deepstack_language_cross_attn"— inject inside LM decoder blocks (cross-attention)

The legacy mode ("lam_add" or any non-"deepstack" string) uses the existing
`VGGTMerger` + weighted-sum path in `modeling_qwen2_5_vl.py` and does NOT
use anything in this file.

DIMENSION LEGEND (used in comments throughout this file)
---------------------------------------------------------
  B               : batch size (number of images in one forward pass; often 1 in JanusVLN)
  N_vis           : number of MERGED visual tokens per image from the Qwen vision tower
                    = (h_patch // spatial_merge_size) * (w_patch // spatial_merge_size)
                    where h_patch = image_H // patch_size (= 14 for Qwen2.5-VL)
  N_geo           : number of geometry tokens per image from VGGT.
                    In vision-level fusion (deepstack_vision_*):
                        N_geo = h_patch * w_patch  (one per unmerged patch)
                    In language-level fusion (deepstack_language_*):
                        N_geo = N_vis * spatial_merge_size^2  (unmerged tokens in groups)
  m               : spatial_merge_size (= 2 for Qwen2.5-VL default)
  m^2             : spatial_merge_unit = m*m = 4 (tokens per merged patch position)
  vis_C           : vision token feature dimension (= 1280 for Qwen2.5-VL ViT-L)
  geo_C           : VGGT token feature dimension (= 2048 for VGGT ViT-L aggregator)
  lang_C          : language model hidden size (= 3584 for Qwen2.5-VL 7B)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MultiLayerFeatureFusionConfig:
    """
    Configuration for the deepstack multi-layer feature fusion module.

    Fields
    ------
    fusion_method : str
        One of the four deepstack modes listed in the file docstring.
        Also accepts "lam_add" (legacy; this module is not instantiated then).

    vis_hidden_size : int
        Feature dimension of the Qwen vision tower (vis_C).
        Qwen2.5-VL ViT-L uses 1280.

    geo_hidden_size : int
        Feature dimension of the geometry encoder (geo_C).
        VGGT aggregator output is 2048.

    lang_hidden_size : int
        Feature dimension of the language model hidden states (lang_C).
        Qwen2.5-VL 7B uses 3584.

    geometry_fusion_layers : List[int]
        Which vision-tower OR language-decoder layer indices to inject geometry
        features into.  The same index may appear multiple times (one entry per
        geometry encoder layer mapped to that position).
        Example: [8, 16, 24]  →  inject at vision blocks 8, 16, 24.

    pos_encoding_type : str
        Controls whether to add 2D sinusoidal positional embeddings to queries
        and keys before cross-attention.
        "none"     : no positional embeddings (faster, slightly worse alignment)
        "sincos2d" : 2D sin/cos from position_utils.get_2d_sincos_pos_embed

    spatial_merge_size : int
        The Qwen ViT spatial merge factor m (default 2).  Used to compute the
        number of tokens in each merged group (m^2 = 4) for language-level fusion.

    num_heads : int
        Number of attention heads in CrossAttentionBlock.

    dropout : float
        Dropout rate inside CrossAttentionBlock's attention and MLP.

    include_camera_token : bool
        If True, the VGGT aggregator's first (camera) token is prepended to the
        geometry tokens passed to language-level cross-attention.  The camera token
        carries holistic scene pose information.
    """
    fusion_method: str = "deepstack_vision_add"
    vis_hidden_size: int = 1280       # Qwen2.5-VL ViT-L hidden size
    geo_hidden_size: int = 2048       # VGGT aggregator output size
    lang_hidden_size: int = 3584      # Qwen2.5-VL 7B LM hidden size
    geometry_fusion_layers: List[int] = field(default_factory=lambda: [])
    pos_encoding_type: str = "none"
    spatial_merge_size: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    include_camera_token: bool = False


# ---------------------------------------------------------------------------
# Cross-attention block (shared by vision and language cross-attn modes)
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    A single transformer-style cross-attention block where 2D visual tokens are
    the queries and 3D geometry tokens are the keys and values.

    Architecture (follows a standard "pre-norm" transformer decoder block):

        query  = LayerNorm(features_2d)  +  pos_embed_query   (optional)
        key    = LayerNorm(features_3d)  +  pos_embed_key     (optional)
        value  = LayerNorm(features_3d)
        attn   = MultiheadAttention(query, key, value)
        x      = features_2d + attn                          ← first residual
        x      = x + MLP(LayerNorm(x))                       ← second residual

    WHY CROSS-ATTENTION?
        Additive fusion (`features_2d + f(features_3d)`) applies the SAME geometry
        correction to every visual token.  Cross-attention lets each visual token
        selectively attend to the most relevant geometry tokens, learning which
        geometric cues matter for which patch positions.

    WHY THREE SEPARATE LAYER-NORMS FOR Q / K / V?
        In mixed-precision training (bf16), LayerNorm output may differ slightly
        for Q, K, V if they come from different-precision inputs.  Using three
        independent norms prevents the attention logits from being dominated by
        scale differences between the two feature streams.

    Args:
        hidden_size (int):
            Common feature dimension after projection.  For deepstack_vision_*
            this equals vis_hidden_size (1280).  For deepstack_language_* this
            equals lang_hidden_size (3584).
        num_heads (int):  Number of attention heads.
        dropout (float):  Dropout probability.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # Three separate LayerNorms: one for queries, one for keys, one for values.
        # All operate on `hidden_size`-dimensional vectors.
        self.norm1_query = nn.LayerNorm(hidden_size)   # normalises features_2d before Q
        self.norm1_key   = nn.LayerNorm(hidden_size)   # normalises features_3d before K
        self.norm1_value = nn.LayerNorm(hidden_size)   # normalises features_3d before V
        self.norm2       = nn.LayerNorm(hidden_size)   # normalises combined x before MLP

        # Multi-head cross-attention.
        # batch_first=True means input shape is [B, SeqLen, C] (not [SeqLen, B, C]).
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Position-wise MLP: expand → GELU → contract (standard 4× expansion).
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        features_2d: torch.Tensor,                              # [B, N_q, C]
        features_3d: torch.Tensor,                              # [B, N_k, C]
        pos_embed_query: Optional[torch.Tensor] = None,         # [B, N_q, C] or None
        pos_embed_key: Optional[torch.Tensor] = None,           # [B, N_k, C] or None
    ) -> torch.Tensor:
        """
        Args:
            features_2d : Visual queries.   Shape [B, N_q, C].
                In vision-level fusion: B=n_images, N_q=N_vis (merged patches), C=vis_C.
                In language-level fusion: B=1, N_q=N_img_tokens, C=lang_C.
            features_3d : Geometry keys/values.  Shape [B, N_k, C].
                After projection, geo features are also in C dimensions.
            pos_embed_query : Optional 2D sin/cos for queries. Shape [B, N_q, C].
                Added to Q before attention so the model knows each patch's position.
            pos_embed_key : Optional 2D sin/cos for keys. Shape [B, N_k, C].
                Added to K so the geometry tokens also carry spatial position info.

        Returns:
            Fused features of shape [B, N_q, C].
            The output has the same shape as features_2d — it can be written back
            to the hidden states in-place.
        """
        # --- Pre-normalise and add optional positional embeddings ---
        query = self.norm1_query(features_2d)  # [B, N_q, C]
        key   = self.norm1_key(features_3d)    # [B, N_k, C]
        value = self.norm1_value(features_3d)  # [B, N_k, C]

        # LayerNorm may upcast to fp32 under autocast; cast back to the source dtype.
        query = query.to(features_2d.dtype)
        key   = key.to(features_3d.dtype)
        value = value.to(features_3d.dtype)

        # Add externally computed positional embeddings (from position_utils.py).
        # WHY ADD to Q AND K but not V?
        #   The positional signal guides WHICH tokens attend to WHICH other tokens
        #   (through the QK dot-product similarity).  The values carry the actual
        #   content to be aggregated, so adding position there would corrupt the
        #   geometric feature signal.
        if pos_embed_query is not None:
            query = query + pos_embed_query.to(dtype=query.dtype, device=query.device)
        if pos_embed_key is not None:
            key = key + pos_embed_key.to(dtype=key.dtype, device=key.device)

        # --- Cross-attention ---
        # Each of the N_q visual tokens attends over all N_k geometry tokens.
        # attn_output shape: [B, N_q, C]
        attn_output, _ = self.cross_attention(query, key, value)

        # --- First residual: fuse cross-attention result into visual features ---
        # The residual ensures that if the geometry signal is uninformative, the
        # visual features are unchanged (the attention output can learn to be zero).
        x = features_2d + attn_output  # [B, N_q, C]

        # --- Second residual: MLP refinement ---
        # Standard transformer decoder block pattern.
        mlp_output = self.mlp(self.norm2(x))  # [B, N_q, C]
        x = x + mlp_output                    # [B, N_q, C]

        return x


# ---------------------------------------------------------------------------
# Multi-layer feature fusion module
# ---------------------------------------------------------------------------

class MultiLayerFeatureFusionModule(nn.Module):
    """
    Manages per-layer fusion blocks for deepstack 2D+3D feature injection.

    This module holds one or more learnable fusion blocks, each mapped to a
    specific layer index of the vision tower or language model decoder.
    During the forward pass through the vision/language model, the parent
    model calls `fusion_module(hidden_states, geo_features, layer_num, ...)` at
    the appropriate layer.

    DESIGN: ModuleDict keyed by layer index
    ----------------------------------------
    `self.fusion_layers` is a `nn.ModuleDict` where keys are string layer indices
    and values are `nn.ModuleList` of fusion blocks at that layer.  The list allows
    MULTIPLE geometry encoder layers to feed into the SAME vision/LM layer:

        geometry_encoder_layers = [-4, -2]   # extract VGGT layers -4 and -2
        geometry_fusion_layers  = [12, 12]   # both inject into vision block 12

    Results in: self.fusion_layers["12"] = ModuleList([block_for_-4, block_for_-2])

    ZERO-INIT FOR STABLE TRAINING
    ------------------------------
    For "deepstack_language_add", the last linear in each geo_mlp is zero-initialised.
    This means at the start of training:
        geo_output = geo_mlp(geo_features) = 0
        hidden_states = hidden_states + 0  =  hidden_states  (unchanged)
    The model starts as if no geometry fusion exists and gradually learns to use it.
    Without zero-init, random geometry features would immediately corrupt the
    pretrained LM representations.

    Args:
        config (MultiLayerFeatureFusionConfig): see dataclass above.
    """

    def __init__(self, config: MultiLayerFeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method   = config.fusion_method
        self.vis_hidden_size = config.vis_hidden_size
        self.geo_hidden_size = config.geo_hidden_size
        self.lang_hidden_size = config.lang_hidden_size

        # Build a ModuleList per unique fusion layer index.
        # Duplicate indices → multiple blocks at that layer.
        self.fusion_layers: nn.ModuleDict = nn.ModuleDict()
        for layer_num in config.geometry_fusion_layers:
            key = str(layer_num)
            if key not in self.fusion_layers:
                self.fusion_layers[key] = nn.ModuleList()
            self.fusion_layers[key].append(self._build_fusion_block())

        # Zero-initialise the residual branches to start as identity.
        # This applies only to "deepstack_language_add" where the output is added
        # directly to LM hidden states (which are very sensitive to perturbation).
        self._zero_init_residual_branches()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_fusion_layer(self, layer_idx: int) -> nn.ModuleList:
        """Return the ModuleList of fusion blocks for a given layer index."""
        return self.fusion_layers[str(layer_idx)]

    # ------------------------------------------------------------------
    # Block construction
    # ------------------------------------------------------------------

    def _build_fusion_block(self) -> nn.Module:
        """
        Build one fusion block according to `self.fusion_method`.

        Four architectures, explained below.
        """
        # Lazy import to avoid circular dependency:
        # modeling_qwen2_5_vl.py defines Qwen2RMSNorm and imports from this file.
        try:
            from .modeling_qwen2_5_vl import Qwen2RMSNorm
        except ImportError:
            Qwen2RMSNorm = nn.LayerNorm  # fallback

        # ---------------------------------------------------------------
        # Mode 1: deepstack_vision_add
        # ---------------------------------------------------------------
        # Inject geometry into VISION TOWER hidden states via addition.
        #
        # Architecture:
        #   RMSNorm(geo_C) → Linear(geo_C → geo_C*2) → GELU → Linear(geo_C*2 → vis_C)
        #   vision_states += geo_projected
        #
        # Dimensions:
        #   Input  : [B, N_vis, geo_C]  = [B, N_vis, 2048]
        #   Output : [B, N_vis, vis_C]  = [B, N_vis, 1280]
        #
        # WHY geo_C*2 as intermediate?
        #   VGGT features (2048-dim) carry rich geometric information.  A 2× expansion
        #   lets the MLP learn nonlinear combinations before projecting down to 1280.
        if self.fusion_method == "deepstack_vision_add":
            return nn.Sequential(
                Qwen2RMSNorm(self.geo_hidden_size, eps=1e-6),
                nn.Linear(self.geo_hidden_size, self.geo_hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.geo_hidden_size * 2, self.vis_hidden_size),
            )

        # ---------------------------------------------------------------
        # Mode 2: deepstack_vision_cross_attn
        # ---------------------------------------------------------------
        # Inject geometry into VISION TOWER hidden states via cross-attention.
        #
        # Architecture:
        #   geo_proj : LayerNorm(geo_C) → Linear(geo_C → vis_C)   [project to vis_C first]
        #   cross_attn: CrossAttentionBlock(vis_C)
        #       query = vision_states   [B, N_vis, vis_C]
        #       key   = geo_proj(geo)   [B, N_geo, vis_C]
        #       value = geo_proj(geo)   [B, N_geo, vis_C]
        #
        # WHY project geo to vis_C BEFORE attention?
        #   MultiheadAttention requires Q, K, V to have the same embed_dim.
        #   geo_C (2048) ≠ vis_C (1280), so we project first with a learned linear.
        #   The projection also acts as a bottleneck that filters which geometry
        #   information is relevant to the vision feature space.
        elif self.fusion_method == "deepstack_vision_cross_attn":
            return nn.ModuleDict({
                "geo_proj": nn.Sequential(
                    nn.LayerNorm(self.geo_hidden_size),           # normalise geo features
                    nn.Linear(self.geo_hidden_size, self.vis_hidden_size),  # [geo_C → vis_C]
                ),
                "cross_attn": CrossAttentionBlock(
                    self.vis_hidden_size,
                    self.config.num_heads,
                    self.config.dropout,
                ),
            })

        # ---------------------------------------------------------------
        # Mode 3: deepstack_language_add
        # ---------------------------------------------------------------
        # Inject geometry into LANGUAGE MODEL hidden states via addition,
        # applied only at image-token positions.
        #
        # Architecture:
        #   RMSNorm(geo_C) → flatten m^2 tokens → Linear(geo_C*m^2 → 4096) → GELU → Linear(4096 → lang_C)
        #   image_token_hidden_states += geo_projected
        #
        # WHY flatten m^2 tokens BEFORE the MLP?
        #   At language level, each LM image token corresponds to one MERGED patch
        #   (spatial_merge_size^2 = 4 raw patches).  We concatenate those 4 raw geo
        #   tokens into a single vector before projecting, so the MLP can learn
        #   spatial relationships within each merged group.
        #
        # Dimensions:
        #   geo input : [B, N_vis * m^2, geo_C]  = [B, N_vis*4, 2048]
        #   after RMSNorm+flatten: [B*N_vis, geo_C*m^2]  = [B*N_vis, 8192]
        #   after geo_mlp:  [B*N_vis, lang_C]  = [B*N_vis, 3584]
        #   LM image tokens: [B*N_vis, lang_C]
        elif self.fusion_method == "deepstack_language_add":
            m = self.config.spatial_merge_size          # = 2
            return nn.ModuleDict({
                "geo_ln": Qwen2RMSNorm(self.geo_hidden_size, eps=1e-6),
                "geo_mlp": nn.Sequential(
                    nn.Linear(self.geo_hidden_size * m * m, 4096),
                    nn.GELU(),
                    nn.Linear(4096, self.lang_hidden_size),
                ),
            })

        # ---------------------------------------------------------------
        # Mode 4: deepstack_language_cross_attn
        # ---------------------------------------------------------------
        # Inject geometry into LANGUAGE MODEL hidden states via cross-attention.
        # Optionally includes the VGGT camera token for global scene context.
        #
        # Architecture:
        #   geo_patch_proj: RMSNorm → flatten m^2 → Linear(geo_C*m^2 → 4096) → Linear(4096 → lang_C)
        #   cam_proj:       RMSNorm → Linear(geo_C → lang_C)   [optional camera token path]
        #   cross_attn:     CrossAttentionBlock(lang_C)
        #       query = LM image tokens  [B, N_vis, lang_C]
        #       key/value = projected geo tokens  [B, N_vis (+ 1 cam), lang_C]
        #
        # WHY a separate cam_proj?
        #   The camera token is a single holistic scene descriptor (not a local patch),
        #   so it should NOT be processed by the patch-flattening MLP.  A dedicated
        #   linear projection handles it separately and prepends it to the key sequence.
        elif self.fusion_method == "deepstack_language_cross_attn":
            m = self.config.spatial_merge_size
            return nn.ModuleDict({
                "geo_ln":  Qwen2RMSNorm(self.geo_hidden_size, eps=1e-6),
                "geo_mlp": nn.Sequential(
                    nn.Linear(self.geo_hidden_size * m * m, 4096),
                    nn.GELU(),
                    nn.Linear(4096, self.lang_hidden_size),
                ),
                "cam_proj": nn.Sequential(
                    Qwen2RMSNorm(self.geo_hidden_size, eps=1e-6),
                    nn.Linear(self.geo_hidden_size, lang_hidden_size := self.lang_hidden_size),
                    nn.GELU(),
                    nn.Linear(lang_hidden_size, self.lang_hidden_size),
                ),
                "cross_attn": CrossAttentionBlock(
                    self.lang_hidden_size,
                    self.config.num_heads,
                    self.config.dropout,
                ),
            })

        else:
            raise ValueError(
                f"Unknown fusion_method '{self.fusion_method}'. "
                "Expected one of: deepstack_vision_add, deepstack_vision_cross_attn, "
                "deepstack_language_add, deepstack_language_cross_attn."
            )

    # ------------------------------------------------------------------
    # Zero-init for stable training start
    # ------------------------------------------------------------------

    def _zero_init_residual_branches(self) -> None:
        """
        Zero-initialise the final linear layer of every geo_mlp in
        "deepstack_language_add" fusion blocks.

        After zero-init, at t=0:
            geo_mlp(any_input) = 0
            LM_hidden += 0  → LM is unchanged (identity)

        This prevents geometry noise from corrupting pretrained LM weights
        at the start of fine-tuning.  The language decoder gradually learns
        to leverage geometry as training proceeds.
        """
        if self.config.fusion_method != "deepstack_language_add":
            return
        for mod_list in self.fusion_layers.values():
            for block in mod_list:
                self._zero_last_linear(block["geo_mlp"])

    @staticmethod
    def _zero_last_linear(module: nn.Module) -> None:
        """Find the last nn.Linear in `module` and zero its weight and bias."""
        for layer in reversed(list(module.modules())):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                return

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
        layer_num: int,
        vis_pos_embed: Optional[torch.Tensor] = None,
        geo_pos_embed: Optional[torch.Tensor] = None,
        fusion_layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply the fusion block for `layer_num` to combine 2D and 3D features.

        Args:
            features_2d (torch.Tensor):
                The hidden states to be updated.
                - deepstack_vision_*  : [B, N_vis, vis_C]   (vision block output)
                - deepstack_language_*: [B*N_vis, lang_C]   (LM image-token slice)

            features_3d (torch.Tensor):
                Geometry encoder features aligned to features_2d.
                - deepstack_vision_add/cross_attn: [B, N_vis, geo_C]
                    (one geo token per merged visual patch; same N_vis count)
                - deepstack_language_add: [B, N_vis * m^2, geo_C]
                    (m^2 raw geo tokens per merged visual patch, to be flattened)
                - deepstack_language_cross_attn: [B, N_vis * m^2 (+ 1 cam), geo_C]

            layer_num (int):
                Index of the current vision/language layer being processed.
                Used to look up the correct fusion block in self.fusion_layers.

            vis_pos_embed (torch.Tensor, optional):
                2D sincos positional embeddings for 2D queries. [B, N_vis, vis_C or lang_C].

            geo_pos_embed (torch.Tensor, optional):
                2D sincos positional embeddings for 3D keys.   [B, N_geo, vis_C or lang_C].

            fusion_layer_idx (int, optional):
                When multiple geometry encoder layers feed into the same fusion layer,
                select a specific block (0-indexed).  None = process all blocks.

        Returns:
            torch.Tensor: Updated features_2d (same shape as input features_2d).
        """
        fusion_block_list = self.get_fusion_layer(layer_num)

        # Normalise features_3d to a list for uniform handling
        features_3d_list: List[torch.Tensor]
        if isinstance(features_3d, (list, tuple)):
            features_3d_list = list(features_3d)
        else:
            features_3d_list = [features_3d]

        # Determine which block(s) to apply
        if fusion_layer_idx is not None:
            # Caller selects a specific block (used in the vision-tower loop)
            blocks_to_apply = [fusion_block_list[fusion_layer_idx]]
            feats_to_apply  = [features_3d_list[fusion_layer_idx]]
        else:
            # Apply all blocks for this layer in order
            n_blocks = len(fusion_block_list)
            n_feats  = len(features_3d_list)
            if n_feats == 1:
                # Broadcast single geometry feature to all blocks
                feats_to_apply = features_3d_list * n_blocks
            elif n_feats == n_blocks:
                feats_to_apply = features_3d_list
            else:
                raise ValueError(
                    f"Layer {layer_num}: {n_feats} geometry tensors but {n_blocks} "
                    f"fusion blocks. Expected 1 (broadcast) or {n_blocks} (one-to-one)."
                )
            blocks_to_apply = list(fusion_block_list)

        # Apply each (block, geo_feature) pair sequentially
        for block, geo_feats in zip(blocks_to_apply, feats_to_apply):
            features_2d = self._apply_block(block, features_2d, geo_feats, vis_pos_embed, geo_pos_embed)

        return features_2d

    @staticmethod
    def _tile_geo_rows_to_vision_count(
        geo_feats: torch.Tensor,
        n_vision_tokens: int,
    ) -> torch.Tensor:
        """Repeat per-frame geometry rows to match total LM image-token count.

        JanusVLN sequences can contain multiple image placeholders (history frames)
        while VGGT geometry is collected for a single representative frame.  The legacy
        lam_add path tiles 3D features the same way before additive fusion.
        """
        n_geo = geo_feats.shape[0]
        if n_geo == n_vision_tokens:
            return geo_feats
        if n_vision_tokens % n_geo != 0:
            raise ValueError(
                f"Cannot tile {n_geo} geometry tokens to {n_vision_tokens} vision tokens. "
                "Expected vision token count to be an integer multiple of per-frame geometry."
            )
        return geo_feats.repeat(n_vision_tokens // n_geo, 1)

    @staticmethod
    def _tile_geo_tokens_to_vision_count(
        geo_feats: torch.Tensor,
        n_vision_tokens: int,
    ) -> torch.Tensor:
        """Repeat per-frame geometry along the token dimension [B, N, C]."""
        n_geo = geo_feats.shape[1]
        if n_geo == n_vision_tokens:
            return geo_feats
        if n_vision_tokens % n_geo != 0:
            raise ValueError(
                f"Cannot tile {n_geo} geometry tokens to {n_vision_tokens} vision tokens. "
                "Expected vision token count to be an integer multiple of per-frame geometry."
            )
        return geo_feats.repeat(1, n_vision_tokens // n_geo, 1)

    def _apply_block(
        self,
        block: nn.Module,
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
        vis_pos_embed: Optional[torch.Tensor],
        geo_pos_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply a single fusion block (dispatched by fusion_method)."""

        if self.fusion_method == "deepstack_vision_add":
            # block : nn.Sequential  [geo_C → vis_C MLP]
            # features_3d : [B, N_vis, geo_C]
            # block output: [B, N_vis, vis_C]  (same shape as features_2d)
            geo_projected = block(features_3d)           # [B, N_vis, vis_C]
            assert features_2d.shape == geo_projected.shape, (
                f"deepstack_vision_add shape mismatch: "
                f"features_2d={features_2d.shape}, geo_projected={geo_projected.shape}"
            )
            return features_2d + geo_projected

        elif self.fusion_method == "deepstack_vision_cross_attn":
            # block : ModuleDict with "geo_proj" and "cross_attn"
            # features_3d : [B, N_geo, geo_C]  →  project to [B, N_geo, vis_C]
            geo_projected = block["geo_proj"](features_3d)    # [B, N_geo, vis_C]
            return block["cross_attn"](
                features_2d,         # query  [B, N_vis, vis_C]
                geo_projected,       # key/v  [B, N_geo, vis_C]
                vis_pos_embed,       # pos for Q  [B, N_vis, vis_C] or None
                geo_pos_embed,       # pos for K  [B, N_geo, vis_C] or None
            )  # [B, N_vis, vis_C]

        elif self.fusion_method == "deepstack_language_add":
            # block : ModuleDict with "geo_ln" and "geo_mlp"
            # features_3d : [B, N_vis * m^2, geo_C]
            # Flatten each m^2 group, project to lang_C, add to LM hidden states.
            m = self.config.spatial_merge_size
            geo_feats = block["geo_ln"](features_3d)        # [B, N_vis*m^2, geo_C]
            # Flatten m^2 tokens per position: [B*N_vis, geo_C * m^2]
            geo_feats = geo_feats.reshape(-1, self.geo_hidden_size * m * m)
            geo_feats = block["geo_mlp"](geo_feats)         # [B*N_vis_geo, lang_C]
            # features_2d may include multiple image groups in the LM sequence.
            geo_feats = self._tile_geo_rows_to_vision_count(geo_feats, features_2d.shape[0])
            return features_2d + geo_feats

        elif self.fusion_method == "deepstack_language_cross_attn":
            # block : ModuleDict with "geo_ln", "geo_mlp", "cam_proj", "cross_attn"
            m = self.config.spatial_merge_size
            geo_feats = features_3d     # [B, N_tokens, geo_C]
            # N_tokens = N_vis*m^2  or  N_vis*m^2 + 1 (with camera token)

            B = geo_feats.shape[0]
            n_vision_tokens = features_2d.shape[0] // B
            n_vis_geo = (geo_feats.shape[1] - 1) // (m * m) if (
                geo_feats.shape[1] % (m * m) == 1
            ) else geo_feats.shape[1] // (m * m)
            has_camera_token = geo_feats.shape[1] == n_vis_geo * m * m + 1

            if not has_camera_token:
                geo_feats = block["geo_ln"](geo_feats)   # [B, N_vis_geo*m^2, geo_C]
                geo_feats = geo_feats.reshape(-1, self.geo_hidden_size * m * m)
                geo_feats = block["geo_mlp"](geo_feats)  # [B*N_vis_geo, lang_C]
                geo_feats = geo_feats.reshape(B, n_vis_geo, -1)
                geo_feats = self._tile_geo_tokens_to_vision_count(geo_feats, n_vision_tokens)
            else:
                cam_token    = geo_feats[:, 0:1, :]          # [B, 1, geo_C]
                patch_tokens = geo_feats[:, 1:, :]           # [B, N_vis_geo*m^2, geo_C]

                cam_projected = block["cam_proj"](cam_token) # [B, 1, lang_C]

                patch_tokens = block["geo_ln"](patch_tokens) # [B, N_vis_geo*m^2, geo_C]
                patch_tokens = patch_tokens.reshape(-1, self.geo_hidden_size * m * m)
                patch_tokens = block["geo_mlp"](patch_tokens) # [B*N_vis_geo, lang_C]
                patch_tokens = patch_tokens.reshape(B, n_vis_geo, -1)
                patch_tokens = self._tile_geo_tokens_to_vision_count(patch_tokens, n_vision_tokens)

                geo_feats = torch.cat([cam_projected, patch_tokens], dim=1)  # [B, 1+N_vis, lang_C]

            # Cross-attention: LM image tokens attend over projected geo tokens
            features_2d_2d = features_2d.reshape(B, n_vision_tokens, -1)  # [B, N_vis, lang_C]
            features_2d_2d = block["cross_attn"](
                features_2d_2d,   # query  [B, N_vis, lang_C]
                geo_feats,        # k/v    [B, N_vis(+1), lang_C]
                vis_pos_embed,
                geo_pos_embed,
            )  # [B, N_vis, lang_C]
            return features_2d_2d.reshape(B * n_vision_tokens, -1)  # [B*N_vis, lang_C]

        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")
