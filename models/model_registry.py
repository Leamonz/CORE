from timm.models import register_model

from models.co_net import STCrossTransformer
from models.spatial_expert import SpatialTransformer
from models.temporal_expert import TemporalTransformer


@register_model
def cocast(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = STCrossTransformer(
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        composition=False,
        input_emb="cocast",
        aux_loss=True,
        **kwargs,
    )
    return model


@register_model
def cocast_input_difference(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = STCrossTransformer(
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        composition=False,
        input_emb=None,
        aux_loss=False,
        **kwargs,
    )
    return model


@register_model
def cocast_emb(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = STCrossTransformer(
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        composition=False,
        input_emb="cocast",
        aux_loss=False,
        **kwargs,
    )
    return model


@register_model
def cocast_emb_climode(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = STCrossTransformer(
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        composition=False,
        input_emb="climode",
        aux_loss=False,
        **kwargs,
    )
    return model


@register_model
def cocast_no_cross(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = STCrossTransformer(
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        composition=False,
        input_emb="cocast",
        aux_loss=False,
        cross=False,
        **kwargs,
    )
    return model


@register_model
def cocast_spatio_emb(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = STCrossTransformer(
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        composition=False,
        input_emb="spatio",
        aux_loss=False,
        cross=True,
        **kwargs,
    )
    return model


## Spatial Expert
@register_model
def spatial_expert(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = SpatialTransformer(
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        composition=False,
        input_emb="cocast",
        **kwargs,
    )
    return model


@register_model
def spatial_expert_no_emb(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = SpatialTransformer(
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        composition=False,
        input_emb=None,
        **kwargs,
    )
    return model


@register_model
def spatial_expert_no_mlp(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = SpatialTransformer(
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        composition=False,
        input_emb="cocast",
        output_mlp=False,
        **kwargs,
    )
    return model


@register_model
def spatial_expert_no_emb_mlp(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = SpatialTransformer(
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        composition=False,
        input_emb=None,
        output_mlp=False,
        **kwargs,
    )
    return model


## Temporal Expert
@register_model
def temporal_expert(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = TemporalTransformer(
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        composition=False,
        input_emb="cocast",
        **kwargs,
    )
    return model
