"""
Adapted from https://github.com/berniwal/swin-transformer-pytorch

Custom Swin-Transformer Implementation (Ensures Modification of Architecture)
"""

# Changed for Core
import numpy as np
import torch
from einops import rearrange
from torch import einsum, nn


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, pix_map, **kwargs):
        return self.fn(x, pix_map, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, pix_map, **kwargs):
        return self.fn(self.norm(x), pix_map, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x, pix_map):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size**2, window_size**2)

    if upper_lower:
        mask[-displacement * window_size :, : -displacement * window_size] = float("-inf")
        mask[: -displacement * window_size, -displacement * window_size :] = float("-inf")

    if left_right:
        mask = rearrange(mask, "(h1 w1) (h2 w2) -> h1 w1 h2 w2", h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float("-inf")
        mask[:, :-displacement, :, -displacement:] = float("-inf")
        mask = rearrange(mask, "h1 w1 h2 w2 -> (h1 w1) (h2 w2)")

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(
        np.array([[x, y] for x in range(window_size) for y in range(window_size)])
    )
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        head_dim,
        shifted,
        window_size,
        relative_pos_embedding,
        stage,
        mask_pos_emb,
    ):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim**-0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.stage = stage
        self.sizes = (96, 48, 24, 12)
        self.mask_pos_emb = mask_pos_emb

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(
                create_mask(
                    window_size=window_size,
                    displacement=displacement,
                    upper_lower=True,
                    left_right=False,
                ),
                requires_grad=False,
            )
            self.left_right_mask = nn.Parameter(
                create_mask(
                    window_size=window_size,
                    displacement=displacement,
                    upper_lower=False,
                    left_right=True,
                ),
                requires_grad=False,
            )

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(
                torch.randn(2 * window_size - 1, 2 * window_size - 1)
            )
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size**2, window_size**2))

        # Set learnable embeddings for pixel mask (Only if Selected)
        if self.mask_pos_emb:
            self.pix_embedding = nn.Parameter(torch.randn(2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, pix_map):
        if self.shifted:
            x = self.cyclic_shift(x)

        _, n_h, n_w, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d",
                h=h,
                w_h=self.window_size,
                w_w=self.window_size,
            ),
            qkv,
        )
        dots = einsum("b h w i d, b h w j d -> b h w i j", q, k) * self.scale

        if self.mask_pos_emb:
            # Create the pixel map attention mask
            pix_emb = rearrange(pix_map, "b (h1 h) (w1 w) -> b (h1 w1) (h w)", h1=nw_h, w1=nw_w)
            pix_mask_a = pix_emb.unsqueeze(2)
            pix_mask_b = pix_emb.unsqueeze(-1)
            pix_mask = pix_mask_a * pix_mask_b
            pix_mask[pix_mask == 0] = self.pix_embedding[0]
            pix_mask[pix_mask == 1] = self.pix_embedding[1]

            # Add the mask with (Q.K)
            pix_mask = pix_mask.unsqueeze(1)
            dots += pix_mask

        if self.relative_pos_embedding:
            dots += self.pos_embedding[
                self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]
            ]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] = dots[:, :, -nw_w:] + self.upper_lower_mask
            dots[:, :, nw_w - 1 :: nw_w] = dots[:, :, nw_w - 1 :: nw_w] + self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum("b h w i j, b h w j d -> b h w i d", attn, v)
        out = rearrange(
            out,
            "b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)",
            h=h,
            w_h=self.window_size,
            w_w=self.window_size,
            nw_h=nw_h,
            nw_w=nw_w,
        )
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        head_dim,
        mlp_dim,
        shifted,
        window_size,
        relative_pos_embedding,
        stage,
        mask_pos_emb,
    ):
        super().__init__()
        self.attention_block = Residual(
            PreNorm(
                dim,
                WindowAttention(
                    dim=dim,
                    heads=heads,
                    head_dim=head_dim,
                    shifted=shifted,
                    window_size=window_size,
                    relative_pos_embedding=relative_pos_embedding,
                    stage=stage,
                    mask_pos_emb=mask_pos_emb,
                ),
            )
        )
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x, pix_map):
        x = self.attention_block(x, pix_map)
        x = self.mlp_block(x, pix_map)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(
            kernel_size=downscaling_factor, stride=downscaling_factor, padding=0
        )
        self.linear = nn.Linear(in_channels * downscaling_factor**2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dimension,
        layers,
        downscaling_factor,
        num_heads,
        head_dim,
        window_size,
        relative_pos_embedding,
        stage,
        mask_pos_emb,
    ):
        super().__init__()
        assert (
            layers % 2 == 0
        ), "Stage layers need to be divisible by 2 for regular and shifted block."

        self.patch_partition = PatchMerging(
            in_channels=in_channels,
            out_channels=hidden_dimension,
            downscaling_factor=downscaling_factor,
        )

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(
                nn.ModuleList(
                    [
                        SwinBlock(
                            dim=hidden_dimension,
                            heads=num_heads,
                            head_dim=head_dim,
                            mlp_dim=hidden_dimension * 4,
                            shifted=False,
                            window_size=window_size,
                            relative_pos_embedding=relative_pos_embedding,
                            stage=stage,
                            mask_pos_emb=mask_pos_emb,
                        ),
                        SwinBlock(
                            dim=hidden_dimension,
                            heads=num_heads,
                            head_dim=head_dim,
                            mlp_dim=hidden_dimension * 4,
                            shifted=True,
                            window_size=window_size,
                            relative_pos_embedding=relative_pos_embedding,
                            stage=stage,
                            mask_pos_emb=mask_pos_emb,
                        ),
                    ]
                )
            )

    def forward(self, x, pix_map=None):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x, pix_map)
            x = shifted_block(x, pix_map)
        return x.permute(0, 3, 1, 2)


class SwinTransformer(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim,
        layers,
        heads,
        channels=3,
        num_classes=1000,
        head_dim=32,
        window_size=7,
        downscaling_factors=(4, 2, 2, 2),
        relative_pos_embedding=True,
        mask_pos_emb=1,
    ):
        super().__init__()

        self.stage1 = StageModule(
            in_channels=channels,
            hidden_dimension=hidden_dim,
            layers=layers[0],
            downscaling_factor=downscaling_factors[0],
            num_heads=heads[0],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            stage=1,
            mask_pos_emb=mask_pos_emb,
        )
        self.stage2 = StageModule(
            in_channels=hidden_dim,
            hidden_dimension=hidden_dim * 2,
            layers=layers[1],
            downscaling_factor=downscaling_factors[1],
            num_heads=heads[1],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            stage=2,
            mask_pos_emb=mask_pos_emb,
        )
        self.stage3 = StageModule(
            in_channels=hidden_dim * 2,
            hidden_dimension=hidden_dim * 4,
            layers=layers[2],
            downscaling_factor=downscaling_factors[2],
            num_heads=heads[2],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            stage=3,
            mask_pos_emb=mask_pos_emb,
        )
        self.stage4 = StageModule(
            in_channels=hidden_dim * 4,
            hidden_dimension=hidden_dim * 8,
            layers=layers[3],
            downscaling_factor=downscaling_factors[3],
            num_heads=heads[3],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            stage=4,
            mask_pos_emb=mask_pos_emb,
        )
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(hidden_dim * 8),
        #     nn.Linear(hidden_dim * 8, num_classes)
        # )

    def forward(self, img, pixel_maps):
        hiddens = []
        x = self.stage1(img, pix_map=pixel_maps[0])
        # xf = x.clone().view(*x.shape[:2], -1).permute(0,2,1)
        hiddens.append(x)
        x = self.stage2(x, pix_map=pixel_maps[1])
        hiddens.append(x)
        x = self.stage3(x, pix_map=pixel_maps[2])
        hiddens.append(x)
        x = self.stage4(x, pix_map=pixel_maps[3])
        x = x.view(*x.shape[:2], -1).permute(0, 2, 1)
        hiddens.append(x)
        return x, hiddens


def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_s(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_b(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_l(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def test_swin():
    model = swin_b(window_size=12)
    eg_in = torch.randn(1, 3, 384, 384)
    out = model(eg_in, pixel_maps=[0, 0, 0, 0])
    print(f"OUT: {out}")


# Test the Custom SWIN Transformer:
if __name__ == "__main__":
    test_swin()
