# Pooling methods code based on: https://github.com/filipradenovic/cnnimageretrieval-pytorch

import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME
import math

from models.layers.netvlad import NetVLADLoupe
from models.layers.ZCANorm import ZCANormSVDPIv2 as ZCANormSVDPI
# from models.layers.ZCANorm import ZCANormSVDPI_No_Shrink

def intra_normalization(x):

    x = nn.functional.normalize(x, dim=1)
    x = x.transpose(1, 2).flatten(1)
    return nn.functional.normalize(x, dim=-1)

def L2_normalization(x):

    x = x.transpose(1, 2).flatten(1)
    return nn.functional.normalize(x, dim=-1)

class MAC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.f = ME.MinkowskiGlobalMaxPooling()

    def forward(self, x: ME.SparseTensor):
        x = self.f(x)
        return x.F      # Return (batch_size, n_features) tensor


class SPoC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        x = self.f(x)
        return x.F      # Return (batch_size, n_features) tensor


class GeM(nn.Module):
    def __init__(self, input_dim, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        # This implicitly applies ReLU on x (clamps negative values)
        #temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p), coordinates=x.C)
        temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p),
                               coordinate_manager = x.coordinate_manager,
                               coordinate_map_key = x.coordinate_map_key)
        temp = self.f(temp)             # Apply ME.MinkowskiGlobalAvgPooling
        return temp.F.pow(1./self.p)    # Return (batch_size, n_features) tensor


class NetVLADWrapper(nn.Module):
    def __init__(self, feature_size, output_dim, gating=True):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.net_vlad = NetVLADLoupe(feature_size=feature_size, cluster_size=64, output_dim=output_dim, gating=gating,
                                     add_batch_norm=True)

    def forward(self, x: ME.SparseTensor):
        # x is (batch_size, C, H, W)
        assert x.F.shape[1] == self.feature_size
        features = x.decomposed_features
        # features is a list of (n_points, feature_size) tensors with variable number of points
        batch_size = len(features)
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        # features is (batch_size, n_points, feature_size) tensor padded with zeros

        x = self.net_vlad(features)
        assert x.shape[0] == batch_size
        assert x.shape[1] == self.output_dim
        return x    # Return (batch_size, output_dim) tensor

class MAScore(nn.Module):

    def __init__(self, num_channels, num_clusters, num_heads=4, scale=1):

        super().__init__()
        self.mha = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)
        self.WQ = nn.parameter.Parameter(nn.init.xavier_uniform_(torch.empty(num_channels, num_channels)))
        self.WK = nn.parameter.Parameter(nn.init.xavier_uniform_(torch.empty(num_channels, num_channels)))
        self.WV = nn.parameter.Parameter(nn.init.xavier_uniform_(torch.empty(num_channels, num_channels)))
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)
        self.ff = nn.Sequential(
            nn.Linear(num_channels, 2*num_channels),
            nn.GELU(),
            nn.Linear(2*num_channels, num_channels)
        )
        self.proj = nn.Linear(num_channels, num_clusters)
        self.scale = scale

    def forward(self, x, lengths):
        # Assume x \in B x C x N
        x = x.permute(0, 2, 1) # B x N x C
        q = x @ self.WQ
        k = x @ self.WK
        v = x @ self.WV
        
        # ssmax_scale = self.scale * math.log(x.size(-2))
        # x = self.norm1(x + self.mha(ssmax_scale * q, k, v)[0])
        # x = self.norm2(x + self.ff(x))
        # x = self.proj(x).permute(0, 2, 1) # B x M x N
        # x = torch.softmax(ssmax_scale * x , dim=-1)
        # return x
        seq_list = []
        for i, valid_len in enumerate(lengths):
            x_token = x[i][:valid_len, :]
            q_token = q[i][:valid_len, :]; k_token = k[i][:valid_len, :]; v_token = v[i][:valid_len, :]
            # ssmax_scale = self.scale * math.log(q_token.size(0))
            x_token = self.norm1(x_token + self.mha(q_token, k_token, v_token)[0])
            x_token = self.norm2(x_token + self.ff(x_token))
            x_token = self.proj(x_token)
            x_token = torch.softmax(x_token, dim=0)
            seq_list.append(x_token)
        return torch.nn.utils.rnn.pad_sequence(seq_list, batch_first=True).permute(0, 2, 1)

class VoronoiSecond(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
    """
    def __init__(self, input_dim, output_dim, num_clusters=64, cluster_dim=128, is_sqrt=False) -> None:
        super().__init__()

        self.num_channels = input_dim
        self.metric_dim = output_dim
        self.num_clusters= num_clusters
        self.cluster_dim = cluster_dim
        self.is_sqrt = is_sqrt

        self.proj = nn.Sequential(
            nn.Conv1d(self.num_channels, 256, 1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, self.cluster_dim, 1)
        )
        self.score = nn.Sequential(
            nn.Conv1d(self.num_channels, 256, 1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, self.num_clusters, 1),
        )
        self.cluster_norm = ZCANormSVDPI(self.cluster_dim, self.num_clusters, affine=False)

    def softmax_valid_len(self, x, lengths, dim):
        for i, valid_len in enumerate(lengths):
            x[i, :, :valid_len] = torch.softmax(x[i, :, :valid_len], dim=dim)
            x[i, :, valid_len:] = 0
        return x

    def get_whitenings(self, x: ME.SparseTensor):
        # Transform ME sparse tensor to torch tensor
        assert not self.training
        features = x.decomposed_features

        lengths = [len(feat) for feat in features]
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        features = features.permute(0, 2, 1) # NOTE
        # Simple MLPs to compute features and scores
        f = self.proj(features)  # B x C_out x N
        p = self.score(features) # B x Num_Cluster x N
        # p = torch.softmax(p, dim=-1)  # NOTE
        p = self.softmax_valid_len(p, lengths, dim=-1)
        # Compute Global descriptor
        f = torch.einsum('bcn, bmn -> bcm', f, p) # NOTE
        # # Compute Covariance Matrix
        SIGMA_BASE, SIGMA_RBLW, SIGMA_ZCA, trans_BASE, trans_RBLW, trans_ZCA = self.cluster_norm.get_whitenings(f)
        return SIGMA_BASE, SIGMA_RBLW, SIGMA_ZCA, trans_BASE, trans_RBLW, trans_ZCA

    def get_unnormalized(self, x: ME.SparseTensor):
        # Transform ME sparse tensor to torch tensor
        assert not self.training
        features = x.decomposed_features
        lengths = [len(feat) for feat in features]
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        features = features.permute(0, 2, 1) # NOTE
        # Simple MLPs to compute features and scores
        f = self.proj(features)  # B x C_out x N
        p = self.score(features) # B x Num_Cluster x N
        p = self.softmax_valid_len(p, lengths, dim=-1)
        f = torch.einsum('bcn, bmn -> bcm', f, p) # NOTE
        # # Compute Covariance Matrix
        f = f.flatten(1) # NOTE 1/M for WP, 1/sqrt(M) for Oxford
        return f


    def forward(self, x: ME.SparseTensor):
        # Transform ME sparse tensor to torch tensor
        features = x.decomposed_features

        lengths = [len(feat) for feat in features]
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        features = features.permute(0, 2, 1) # NOTE
        # Simple MLPs to compute features and scores
        f = self.proj(features)  # B x C_out x N
        p = self.score(features) # B x Num_Cluster x N
        p = self.softmax_valid_len(p, lengths, dim=-1)
        f = torch.einsum('bcn, bmn -> bcm', f, p) # NOTE
        # NOTE 1/M for WP, 1/sqrt(M) for Oxford
        if self.is_sqrt:
            f = self.cluster_norm(f).flatten(1) / math.sqrt(self.num_clusters)
        else:
            f = self.cluster_norm(f).flatten(1) / self.num_clusters
        return f

