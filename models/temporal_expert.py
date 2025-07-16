# some codes from CLIP github(https://github.com/openai/CLIP), from VideoMAE github(https://github.com/MCG-NJU/VideoMAE)
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import drop_path, to_2tuple, trunc_normal_
from timm.models import register_model
from collections import OrderedDict
from einops import rearrange
import random


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 400,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        **kwargs,
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Adapter(nn.Module):
    def __init__(self, dim, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        down_dim = int(dim * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(dim, down_dim)
        self.D_fc2 = nn.Linear(down_dim, dim)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        if orig_type == torch.float16:
            ret = super().forward(x)
        elif orig_type == torch.float32:
            ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=16,
        tubelet_size=2,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames // self.tubelet_size)
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        s2t_q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        s2t_q = s2t_q * self.scale
        attn = s2t_q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        num_layer=0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
    ):
        super().__init__()
        self.cross = None
        self.num_layer = num_layer
        self.num_heads = num_heads
        self.scale = 0.5
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.act = act_layer()
        ############################ VMAE MHSA ###########################
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )
        self.T_Adapter = Adapter(dim, skip_connect=True)  # base line 의 경우 True.
        ############################ VMAE FFN ###############################
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.T_MLP_Adapter = Adapter(dim, skip_connect=False)
        #######################################################################
        #########################################################################################

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.clip_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, t_x):
        B = t_x.shape[0]
        ############################################################################
        t_x = t_x + self.T_Adapter(self.attn(self.norm1(t_x)))
        t_xn = self.norm2(t_x)
        t_x = (
            t_x + self.mlp(t_xn) + self.drop_path(self.scale * self.T_MLP_Adapter(t_xn))
        )
        ############################################################################

        return t_x


class TemporalTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        down_ratio=2,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        head_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        init_scale=0.0,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
        composition=False,
        pretrained_cfg=None,
        decoder_depth=2,
        data="carbon_tracker",
        out_chans=1,
        input_emb=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.composition = composition
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.data = data
        self.out_chans = out_chans
        self.input_emb = input_emb

        if self.input_emb is not None:
            if self.data == "carbon_tracker":
                in_chans += 45 if self.input_emb == "cocast" else 35
            else:
                in_chans += 27 if self.input_emb == "cocast" else 21

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
            tubelet_size=self.tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        scale = embed_dim**-0.5

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    num_layer=i,
                )
                for i in range(depth)
            ]
        )

        # self.clip_ln_post = LayerNorm(embed_dim)
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.vmae_fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)
        self._init_adpater_weight()

        # output layer
        self.heads = nn.ModuleList()
        for i in range(decoder_depth):
            self.heads.append(nn.Linear(embed_dim, embed_dim))
            self.heads.append(nn.GELU())
        self.heads.append(
            nn.Linear(
                embed_dim, self.patch_size[0] * self.patch_size[1] * self.out_chans
            )
        )
        for m in self.heads:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal(m.weight)
                m.weight.data.mul_(init_scale)
                m.bias.data.mul_(init_scale)
        self.heads = nn.Sequential(*self.heads)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_adpater_weight(self):
        for n, m in self.blocks.named_modules():
            if "Adapter" in n:
                for n2, m2 in m.named_modules():
                    if "D_fc2" in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
            elif "up" in n:
                for n2, m2 in m.named_modules():
                    if isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0)
                        nn.init.constant_(m2.bias, 0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"clip_temporal_embedding", "pos_embed"}

    def reset_fcnorm(self):
        self.vmae_fc_norm = nn.LayerNorm(self.embed_dim)

    def create_cocast_emb(self, x, t):
        # spatial embedding
        consts = np.load("/gemini/data-3/carbon_tracker_6hrs/constants/constants.npz")
        lat, lon = consts["lat"], consts["lon"]
        lat = torch.from_numpy(lat).type(torch.float32).to(x.device)
        lon = torch.from_numpy(lon).type(torch.float32).to(x.device)
        lat, lon = lat.unsqueeze(0), lon.unsqueeze(0)

        sin_lat_emb, cos_lat_emb = torch.sin(lat), torch.cos(lat)
        sin_lon_emb, cos_lon_emb = torch.sin(lon), torch.cos(lon)
        loc_embed = [cos_lat_emb, sin_lat_emb, cos_lon_emb, sin_lon_emb]
        for i in range(len(loc_embed)):
            loc_embed[i] = (
                loc_embed[i]
                .unsqueeze(1)
                .unsqueeze(1)
                .expand(x.shape[0], 1, x.shape[2], *loc_embed[i].shape[1:])
            )
        lat_embed = loc_embed[:2] + [torch.ones_like(loc_embed[0])]
        lon_embed = loc_embed[2:] + [torch.ones_like(loc_embed[0])]

        # temporal embedding
        input_v = x[:, -1, :, :, :].unsqueeze(1)
        t_emb = t.view(-1, 1)
        t_emb = (
            t_emb.unsqueeze(2)
            .unsqueeze(3)
            .unsqueeze(4)
            .repeat(1, 1, *input_v.shape[2:])
        )
        emb = input_v * t_emb
        if self.data == "carbon_tracker":
            daily_emb = [
                torch.cos(torch.pi * emb / 2),
                torch.sin(torch.pi * emb / 2),
            ]
            yearly_emb = [
                torch.cos(torch.pi * emb / 730),
                torch.sin(torch.pi * emb / 730),
            ]
        else:
            daily_emb = []
            yearly_emb = [
                torch.cos(torch.pi * emb / 6),
                torch.sin(torch.pi * emb / 6),
            ]

        time_embed = daily_emb + yearly_emb
        time_embed += [torch.ones_like(time_embed[0])]

        input_embed = []
        for lat_emb in lat_embed:
            for lon_emb in lon_embed:
                for t_emb in time_embed:
                    input_embed.append(lat_emb * lon_emb * t_emb)

        return torch.cat(input_embed, dim=1)

    def create_climode_emb(self, x, t):
        # location embedding
        consts = np.load("/gemini/data-3/carbon_tracker_6hrs/constants/constants.npz")
        lat, lon = consts["lat"], consts["lon"]
        lat = torch.from_numpy(lat).type(torch.float32).to(x.device)
        lon = torch.from_numpy(lon).type(torch.float32).to(x.device)
        lat, lon = lat.unsqueeze(0), lon.unsqueeze(0)

        sin_lat, cos_lat = torch.sin(lat), torch.cos(lat)
        sin_lon, cos_lon = torch.sin(lon), torch.cos(lon)
        sin_cos_emb = sin_lat * cos_lon
        sin_sin_emb = sin_lat * sin_lon

        loc_embed = [
            sin_lat,
            cos_lat,
            sin_lon,
            cos_lon,
            sin_cos_emb,
            sin_sin_emb,
            torch.ones_like(sin_lat),
        ]
        for i in range(len(loc_embed)):
            loc_embed[i] = (
                loc_embed[i]
                .unsqueeze(1)
                .unsqueeze(1)
                .expand(x.shape[0], 1, x.shape[2], *loc_embed[i].shape[1:])
            )

        # day & season embedding
        t_emb = t.view(-1, 1)
        t_emb = (
            t_emb.unsqueeze(2)
            .unsqueeze(3)
            .unsqueeze(4)
            .repeat(1, 1, *input_v.shape[2:])
        )
        if self.data == "carbon_tracker":
            daily_emb = [
                torch.cos(torch.pi * t_emb / 2),
                torch.sin(torch.pi * t_emb / 2),
            ]
            yearly_emb = [
                torch.cos(torch.pi * t_emb / 730),
                torch.sin(torch.pi * t_emb / 730),
            ]
        else:
            daily_emb = []
            yearly_emb = [
                torch.cos(torch.pi * t_emb / 6),
                torch.sin(torch.pi * t_emb / 6),
            ]

        time_embed = daily_emb + yearly_emb + [torch.ones_like(t_emb)]

        input_embed = []
        for loc_e in loc_embed:
            for t_e in time_embed:
                emb = loc_e * t_e
                input_embed.append(emb.to(x.device))

        return torch.cat(input_embed, dim=1)

    def forward_features(self, x):
        B, C, T, H, W = x.shape
        ######################## VMAE spatial path #########################
        t_x = self.patch_embed(x)

        if self.pos_embed is not None:
            t_x = (
                t_x
                + self.pos_embed.expand(B, -1, -1)
                .type_as(t_x)
                .to(t_x.device)
                .clone()
                .detach()
            )
        t_x = self.pos_drop(t_x)
        #####################################################################

        for blk in self.blocks:
            t_x = blk(t_x)
        t_x = self.vmae_fc_norm(t_x)  # all patch avg pooling

        t_x = rearrange(t_x, "b (n t) d -> b t n d", t=T // self.tubelet_size)
        # print(t_x.shape)
        t_x = t_x.mean(dim=1)
        return t_x

    def forward(self, x, t):
        # x (b, c, t, h, w)
        if self.input_emb is not None:
            input_embed = (
                self.create_cocast_emb(x, t)
                if self.input_emb == "cocast"
                else self.create_climode_emb(x.device, t)
            )
            x = torch.cat([input_embed, x], dim=1)

        t_x = self.forward_features(x)
        x = self.heads(t_x)
        x = x.view(x.shape[0], -1).contiguous()
        x = x.view(
            x.shape[0], self.out_chans, self.img_size[0], self.img_size[1]
        ).contiguous()
        return x
