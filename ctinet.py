"""
CTINet: An fNIRS-Informed Cross-Modal Token Interaction Network
for EEG-fNIRS Fusion

Paper authors:
    Xinyu Zhang, Keum-Shik Hong, Guanghao Huang,
    Peng Sun, and Haiqiang Yang

Code implementation:
    Xinyu Zhang

Corresponding author:
    Haiqiang Yang

Core PyTorch implementation of the CTINet architecture.
See README.md for dataset information, usage, and citation details.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def same_pad_3d(x, kernel, stride, dilation=(1, 1, 1)):
    pads = []
    for i in reversed(range(3)):
        dim = x.shape[2 + i]
        k_eff = kernel[i] + (kernel[i] - 1) * (dilation[i] - 1)
        out = math.ceil(dim / stride[i])
        total = max(0, (out - 1) * stride[i] + k_eff - dim)
        pads.extend([total // 2, total - total // 2])
    return F.pad(x, pads, mode="replicate")


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1):
        super().__init__()
        self.kernel = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        )
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.dilation = (
            dilation if isinstance(dilation, tuple) else (dilation,) * 3
        )
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel,
            self.stride,
            padding=0,
            dilation=self.dilation,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ELU()

    def forward(self, x):
        x = same_pad_3d(x, self.kernel, self.stride, self.dilation)
        return self.act(self.bn(self.conv(x)))


def pearson_correlation(x, y):
    x = x.flatten(1)
    y = y.flatten(1)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    numerator = (x * y).mean(dim=1)
    denominator = x.std(dim=1, unbiased=False) * y.std(
        dim=1, unbiased=False
    ) + 1e-6
    return numerator / denominator


class ChannelWiseHemodynamicModulation(nn.Module):

    def __init__(
        self,
        in_channels,
        lag=11,
        aux_lambda=0.5,
        spatial_lambda=0.05,
    ):
        super().__init__()
        self.lag = lag
        self.spatial_lambda = spatial_lambda
        self.aux_lambda = aux_lambda
        self.pool = nn.Conv3d(
            in_channels,
            1,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1,
        )
        self.channel_gate_raw = nn.Parameter(
            torch.zeros(1, in_channels, 1, 1, 1)
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 2),
        )

    def forward(self, eeg_feat, fnirs_feat, labels=None):
        batch = eeg_feat.size(0)
        if fnirs_feat.size(0) != batch * self.lag:
            raise ValueError(
                f"Expected fNIRS batch {batch * self.lag}, "
                f"got {fnirs_feat.size(0)}"
            )

        # Conv3D -> temporal average -> average over delayed windows -> sigmoid.
        raw = self.pool(fnirs_feat)
        raw = raw.mean(dim=-1)  # (B*lag, 1, H, W)
        raw = raw.reshape(batch, self.lag, 1, raw.size(-2), raw.size(-1))
        aspa = torch.sigmoid(raw.mean(dim=1))  # (B, 1, H, W)

        gate = 2.0 * torch.sigmoid(self.channel_gate_raw)
        emod = eeg_feat * aspa.unsqueeze(-1) * gate
        blend = torch.sigmoid(self.gamma)
        eeg_out = blend * eeg_feat + (1.0 - blend) * emod

        # E-bar_i(h,w) = sqrt(mean over EEG channels and time of E^2 + eps).
        eeg_spatial = torch.sqrt(
            eeg_feat.pow(2).mean(dim=(1, 4)) + 1e-6
        )
        aspa_spatial = aspa.squeeze(1)
        spatial_corr = pearson_correlation(eeg_spatial, aspa_spatial)
        spatial_loss = (1.0 - spatial_corr).mean()

        aux_logits = self.aux_classifier(aspa)
        aux_loss = torch.zeros((), device=eeg_feat.device)
        if self.training and labels is not None:
            aux_loss = F.cross_entropy(aux_logits, labels)

        chm_loss = (
            self.spatial_lambda * spatial_loss
            + self.aux_lambda * aux_loss
        )
        return eeg_out, chm_loss, aspa, aux_logits


class TemporalAttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1)


class MWST(nn.Module):
    """
    Paper MWST.

    Each descriptor has the complete D=512 dimensions.  The same projection
    is shared by mean/std/global-deviation/gradient within one modality.
    """

    def __init__(self, input_dim=512, descriptor_dim=256):
        super().__init__()
        self.projection = nn.Linear(
            input_dim, descriptor_dim, bias=False
        )
        self.input_dim = input_dim
        self.descriptor_dim = descriptor_dim

    def forward(self, sequence, windows):
        batch, time_len, dim = sequence.shape
        if dim != self.input_dim:
            raise ValueError(f"MWST expects {self.input_dim} dims, got {dim}")

        means, stds = [], []
        for index in range(windows):
            start = int(index * time_len / windows)
            end = int((index + 1) * time_len / windows)
            segment = sequence[:, start:end, :]
            means.append(segment.mean(dim=1))
            stds.append(
                segment.std(dim=1, unbiased=False)
                if segment.size(1) > 1
                else torch.zeros_like(segment[:, 0, :])
            )

        means = torch.stack(means, dim=1)
        stds = torch.stack(stds, dim=1)
        global_mean = means.mean(dim=1, keepdim=True)
        global_deviation = means - global_mean
        temporal_gradient = torch.zeros_like(means)
        if windows > 1:
            temporal_gradient[:, 1:, :] = means[:, 1:, :] - means[:, :-1, :]

        descriptors = torch.stack(
            [means, stds, global_deviation, temporal_gradient],
            dim=2,
        )  # (B, K, 4, 512)
        projected = self.projection(descriptors)
        return projected.reshape(batch, windows, 4 * self.descriptor_dim)


class CMIT(nn.Module):

    def __init__(
        self,
        token_dim=1024,
        latent_dim=128,
        heads=4,
        layers=2,
        dropout=0.1,
        max_tokens=4,
    ):
        super().__init__()
        self.eeg_proj = nn.Linear(token_dim, latent_dim, bias=False)
        self.fnirs_proj = nn.Linear(token_dim, latent_dim, bias=False)
        self.eeg_pos = nn.Parameter(torch.zeros(1, max_tokens, latent_dim))
        self.fnirs_pos = nn.Parameter(torch.zeros(1, max_tokens, latent_dim))
        nn.init.trunc_normal_(self.eeg_pos, std=0.02)
        nn.init.trunc_normal_(self.fnirs_pos, std=0.02)

        blocks = []
        for _ in range(layers):
            blocks.append(
                nn.ModuleDict(
                    {
                        "fnirs_to_eeg": nn.MultiheadAttention(
                            latent_dim,
                            heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "eeg_to_fnirs": nn.MultiheadAttention(
                            latent_dim,
                            heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "eeg_ffn": nn.Sequential(
                            nn.Linear(latent_dim, latent_dim * 4),
                            nn.GELU(),
                            nn.Linear(latent_dim * 4, latent_dim),
                        ),
                        "fnirs_ffn": nn.Sequential(
                            nn.Linear(latent_dim, latent_dim * 4),
                            nn.GELU(),
                            nn.Linear(latent_dim * 4, latent_dim),
                        ),
                        "eeg_norm1": nn.LayerNorm(latent_dim),
                        "eeg_norm2": nn.LayerNorm(latent_dim),
                        "fnirs_norm1": nn.LayerNorm(latent_dim),
                        "fnirs_norm2": nn.LayerNorm(latent_dim),
                    }
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.eeg_back = nn.Linear(latent_dim, token_dim)
        self.fnirs_back = nn.Linear(latent_dim, token_dim)

        gate_dim = token_dim // 8
        self.eeg_gate = nn.Sequential(
            nn.Linear(token_dim * 2, gate_dim),
            nn.GELU(),
            nn.Linear(gate_dim, token_dim),
        )
        self.fnirs_gate = nn.Sequential(
            nn.Linear(token_dim * 2, gate_dim),
            nn.GELU(),
            nn.Linear(gate_dim, token_dim),
        )

    def forward(self, eeg_tokens, fnirs_tokens):
        eeg_original = eeg_tokens
        fnirs_original = fnirs_tokens
        eeg = self.eeg_proj(eeg_tokens) + self.eeg_pos[:, : eeg_tokens.size(1)]
        fnirs = self.fnirs_proj(fnirs_tokens) + self.fnirs_pos[
            :, : fnirs_tokens.size(1)
        ]

        for block in self.blocks:
            # Both directions use the same pre-update eeg/fNIRS states.
            eeg_cross, _ = block["fnirs_to_eeg"](
                eeg, fnirs, fnirs
            )
            fnirs_cross, _ = block["eeg_to_fnirs"](
                fnirs, eeg, eeg
            )

            eeg = block["eeg_norm1"](eeg + eeg_cross)
            eeg = block["eeg_norm2"](eeg + block["eeg_ffn"](eeg))
            fnirs = block["fnirs_norm1"](fnirs + fnirs_cross)
            fnirs = block["fnirs_norm2"](
                fnirs + block["fnirs_ffn"](fnirs)
            )

        eeg_update = self.eeg_back(eeg)
        fnirs_update = self.fnirs_back(fnirs)
        eeg_gate = torch.sigmoid(
            self.eeg_gate(torch.cat([eeg_update, eeg_original], dim=-1))
        )
        fnirs_gate = torch.sigmoid(
            self.fnirs_gate(
                torch.cat([fnirs_update, fnirs_original], dim=-1)
            )
        )
        eeg_out = eeg_gate * eeg_update + (1.0 - eeg_gate) * eeg_original
        fnirs_out = (
            fnirs_gate * fnirs_update
            + (1.0 - fnirs_gate) * fnirs_original
        )
        return eeg_out, fnirs_out


class ModalityAdaptiveFusion(nn.Module):
    """MAF with two independent sigmoid modality gates."""

    def __init__(self, dim):
        super().__init__()
        hidden = 64
        self.eeg_score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.fnirs_score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, eeg_feat, fnirs_feat):
        w_eeg = torch.sigmoid(self.eeg_score(eeg_feat))
        w_fnirs = torch.sigmoid(self.fnirs_score(fnirs_feat))
        eeg_fused = w_eeg * eeg_feat + (1.0 - w_eeg) * fnirs_feat
        fnirs_fused = w_fnirs * fnirs_feat + (1.0 - w_fnirs) * eeg_feat
        return torch.cat([eeg_fused, fnirs_fused], dim=1), w_eeg, w_fnirs


class CTINet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        windows=4,
        latent_dim=128,
        cmit_layers=2,
        dropout=0.5,
    ):
        super().__init__()
        self.windows = windows
        self.lag = 11
        self.eeg_conv1 = Conv3dBlock(
            1, 16, kernel_size=(2, 2, 13), stride=(2, 2, 6)
        )
        self.fnirs_conv1 = Conv3dBlock(
            2, 16, kernel_size=(2, 2, 5), stride=(2, 2, 2)
        )
        self.chm = ChannelWiseHemodynamicModulation(16, lag=self.lag)
        self.eeg_conv2 = Conv3dBlock(
            16, 32, kernel_size=(2, 2, 5), stride=(2, 2, 2)
        )
        self.fnirs_conv2 = Conv3dBlock(
            16, 32, kernel_size=(2, 2, 3), stride=(2, 2, 2)
        )
        self.dropout = nn.Dropout(dropout)

        self.feature_dim = 32 * 4 * 4
        self.token_dim = 4 * 256
        self.eeg_mwst = MWST(self.feature_dim, 256)
        self.fnirs_mwst = MWST(self.feature_dim, 256)
        self.cmit = CMIT(
            token_dim=self.token_dim,
            latent_dim=latent_dim,
            heads=4,
            layers=cmit_layers,
            dropout=0.1,
            max_tokens=windows,
        )
        self.eeg_pool = TemporalAttentionPooling(self.token_dim)
        self.fnirs_pool = TemporalAttentionPooling(self.token_dim)
        self.fusion = ModalityAdaptiveFusion(self.token_dim)
        self.classifier = nn.Linear(self.token_dim * 2, num_classes)

    @staticmethod
    def spatial_flatten(feature):
        batch, channels, height, width, time_len = feature.shape
        feature = feature.permute(0, 4, 1, 2, 3).contiguous()
        return feature.reshape(batch, time_len, channels * height * width)

    def forward(self, eeg, fnirs, labels=None, return_intermediate=False):
        batch = eeg.size(0)
        eeg_3d = eeg.permute(0, 4, 1, 2, 3).float()
        fnirs_merged = fnirs.float().reshape(
            batch * self.lag, 2, 16, 16, fnirs.size(-1)
        )

        eeg_feat = self.eeg_conv1(eeg_3d)
        fnirs_feat = self.fnirs_conv1(fnirs_merged)
        eeg_feat, chm_loss, aspa, aux_logits = self.chm(
            eeg_feat, fnirs_feat, labels=labels
        )

        eeg_feat = self.eeg_conv2(eeg_feat)
        fnirs_feat = fnirs_feat.reshape(
            batch, self.lag, *fnirs_feat.shape[1:]
        ).mean(dim=1)
        fnirs_feat = self.fnirs_conv2(fnirs_feat)

        eeg_seq = self.spatial_flatten(self.dropout(eeg_feat))
        fnirs_seq = self.spatial_flatten(self.dropout(fnirs_feat))
        eeg_tokens = self.eeg_mwst(eeg_seq, self.windows)
        fnirs_tokens = self.fnirs_mwst(fnirs_seq, self.windows)
        eeg_tokens, fnirs_tokens = self.cmit(eeg_tokens, fnirs_tokens)

        eeg_embed = self.eeg_pool(eeg_tokens)
        fnirs_embed = self.fnirs_pool(fnirs_tokens)
        fused, w_eeg, w_fnirs = self.fusion(eeg_embed, fnirs_embed)
        fused = F.layer_norm(fused, fused.shape[1:])
        logits = self.classifier(fused)

        result = {
            "class_output": logits,
            "eeg_embed": eeg_embed,
            "fnirs_embed": fnirs_embed,
            "chm_loss": chm_loss,
        }
        if return_intermediate:
            result.update(
                {
                    "chm_attention": aspa,
                    "chm_aux_logits": aux_logits,
                    "fusion_w_eeg": w_eeg,
                    "fusion_w_fnirs": w_fnirs,
                }
            )
        return result
