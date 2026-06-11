import transformers
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_mm_fusion: bool = field(default=True)  # Whether to train deepstack fusion modules

    # JanusVLN-specific VGGT / additive-fusion settings
    vggt_model_path: str = field(default="facebook/VGGT-1B/")
    lam: float = field(default=0.2)
    distill_loss_weight: float = field(default=1.0)
    reference_frame: str = field(default="first")

    # Deepstack multi-layer fusion settings (SpatialStack-style)
    feature_fusion_method: str = field(
        default="lam_add",
        metadata={"help": "Fusion strategy: 'lam_add' (legacy), 'deepstack_vision_add', "
                          "'deepstack_vision_cross_attn', 'deepstack_language_add', "
                          "'deepstack_language_cross_attn'"},
    )
    geometry_fusion_layers: Optional[List[int]] = field(
        default=None,
        metadata={"help": "LM decoder layer indices that receive geometry (deepstack_language_*), "
                          "or vision block indices (deepstack_vision_*). E.g. --geometry_fusion_layers 0 1 2"},
    )
    geometry_encoder_layers: Optional[List[int]] = field(
        default=None,
        metadata={"help": "VGGT aggregator layer indices whose features are extracted for deepstack. "
                          "Must be the same length as geometry_fusion_layers. E.g. --geometry_encoder_layers 11 17 23"},
    )
    pos_encoding_type: str = field(
        default="none",
        metadata={"help": "Positional encoding applied to geometry tokens before cross-attention: "
                          "'none', 'sincos2d'"},
    )
    include_camera_token: bool = field(
        default=False,
        metadata={"help": "Whether to pass the VGGT camera token into the fusion module"},
    )
    fusion_attention_heads: int = field(default=8)
    fusion_dropout: float = field(default=0.1)


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    max_samples: int = field(default=-1)
    shuffle: bool = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
