import torch
import torch.nn as nn


class ClusterNorm1d(nn.Module):

    def __init__(self, cluster_dim, num_clusters, affine=False):

        super().__init__()
        self.cluster_dim  = cluster_dim
        self.num_clusters = num_clusters
        self.affine = affine
        if self.affine:
            self.scale = nn.Parameter(torch.ones(self.cluser_dim))
            self.bias  = nn.Parameter(torch.randn(self.clusre_dim)/self.num_clusters)
        self.register_buffer('mu_track', torch.randn(self.cluster_dim, self.num_clusters))
        self.register_buffer('L_track', torch.randn(self.num_clusters, self.cluster_dim, self.cluster_dim))
        self.register_buffer('EPS', torch.eye(self.cluster_dim) * 1e-4)

    def forward(self, x):

        if not self.training:
            x_mu = self.mu_track.unsqueeze(0)
            x_cnt = (x - x_mu).permute(2, 1, 0)
            # Z = torch.linalg.solve_triangular(self.L_track, x_cnt)
            Z = torch.linalg.solve_triangular(self.L_track, x_cnt, upper=False)
            if self.affine:
                Z = Z * self.scale.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)
            return Z.permute(2, 1, 0)

        # TRAINING
        B = x.size(0)
        x_mu = torch.mean(x, dim=0, keepdim=True) # (B x C x M) -> (1 x C x M)
        x_cnt = x - x_mu
        x_cnt = x_cnt.permute(2, 1, 0) # (B x C x M -> M x C x B)
        cov = torch.bmm(x_cnt, x_cnt.permute(0, 2, 1)) / (B - 1) # M x C x C
        L = torch.linalg.cholesky(cov + self.EPS) # M x C x C
        Z = torch.linalg.solve_triangular(L, x_cnt, upper=False) # M x C x B
        # if not torch.isfinite(cov).all():
        #     print(x)
        #     print(x_cnt)
        #     print(cov)
        # LD, _ = torch.linalg.ldl_factor(cov)
        # D = torch.diag_embed(torch.diagonal(LD, dim1=-2, dim2=-1))
        # L = torch.tril(LD, diagonal=-1) + torch.eye(self.cluster_dim).to(x)
        # L = L @ D.sqrt()
        # Z = torch.linalg.solve_triangular(L, x_cnt, upper=False)
        if self.affine:
            Z = Z * self.scale.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)
        self.mu_track = x_mu.squeeze(0); self.L_track = L
        return Z.permute(2, 1, 0) # B x C x M


class ClusterNorm1dv2(nn.Module):

    def __init__(self, cluster_dim, num_clusters, affine=False):

        super().__init__()
        self.cluster_dim  = cluster_dim
        self.num_clusters = num_clusters
        self.affine = affine
        if self.affine:
            self.scale = nn.Parameter(torch.ones(self.cluser_dim))
            self.bias  = nn.Parameter(torch.randn(self.clusre_dim)/self.num_clusters)
        self.register_buffer('mu_0', torch.zeros(self.cluster_dim, self.num_clusters))
        self.register_buffer('L_0', torch.diag_embed(torch.ones(self.num_clusters, self.cluster_dim)))
        self.register_buffer('EPS', torch.eye(self.cluster_dim))
        self.register_buffer('n_0', torch.ones(1))

    def forward(self, x):

        if not self.training:
            x_mu = self.mu_0.unsqueeze(0)
            x_cnt = (x - x_mu).permute(2, 1, 0)
            Z = torch.linalg.solve_triangular(self.L_0, x_cnt, upper=False)
            if self.affine:
                Z = Z * self.scale.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)
            return Z.permute(2, 1, 0)

        # TRAINING
        B = x.size(0)
        x_mu = torch.mean(x, dim=0, keepdim=True) # (B x C x M) -> (1 x C x M)
        x_cnt = x - x_mu
        x_cnt = x_cnt.permute(2, 1, 0) # (B x C x M -> M x C x B)
        C = torch.bmm(x_cnt, x_cnt.permute(0, 2, 1)) # M x C x C

        new_mu  = (self.n_0 * self.mu_0 + B * x_mu.squeeze(0)) / (self.n_0 + B) # C x M
        x_d     = (x_mu.squeeze(0) - self.mu_0).unsqueeze(0).permute(2, 1, 0)   # M x C x 1
        new_Cov = self.L_0 @ self.L_0.mT * (self.n_0 / (self.n_0 + B)) + \
                  C / (self.n_0 + B) + \
                  x_d @ x_d.mT * self.n_0 * B / ((self.n_0 + B)**2)

        L = torch.linalg.cholesky(new_Cov + self.EPS) # M x C x C
        Z = torch.linalg.solve_triangular(L, (x - new_mu.unsqueeze(0)).permute(2, 1, 0), upper=False) # M x C x B
        if self.affine:
            Z = Z * self.scale.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)
        self.mu_0 = new_mu.clone().detach(); self.L_0 = L.clone().detach(); self.n_0 = self.n_0 + B
        return Z.permute(2, 1, 0) # B x C x M


# ZCA Whitening (Mahalanobis transformation)
# https://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening
class ClusterNorm1dv3(nn.Module):

    def __init__(self, cluster_dim, num_clusters, affine=False, momentum=0.1):

        super().__init__()
        self.cluster_dim  = cluster_dim
        self.num_clusters = num_clusters
        self.affine = affine
        self.momentum = momentum
        if self.affine:
            self.scale = nn.Parameter(torch.ones(self.cluser_dim))
            self.bias  = nn.Parameter(torch.randn(self.clusre_dim)/self.num_clusters)
        self.register_buffer('mu_track', torch.zeros(self.cluster_dim, self.num_clusters))
        self.register_buffer('Std_inv_track', torch.diag_embed(torch.ones(self.num_clusters, self.cluster_dim)))
        self.register_buffer('EPS', torch.ones(1) * 1e-6)
        # self.register_buffer('EPS', torch.eye(self.cluster_dim) * 1e-6)

    def forward(self, x):

        x_mu = self.mu_track.unsqueeze(0)
        x_cnt = (x - x_mu).permute(2, 1, 0)
        Z = self.Std_inv_track @ x_cnt
        if self.affine:
            Z = Z * self.scale.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)

        # TRAINING
        if self.training:
            B = x.size(0)
            x_mu = torch.sum(x, dim=0, keepdim=True) / (B - 1) # (B x C x M) -> (1 x C x M)
            x_cnt = x - x_mu
            x_cnt = x_cnt.permute(2, 1, 0) # (B x C x M -> M x C x B)
            cov = torch.bmm(x_cnt, x_cnt.permute(0, 2, 1)) / (B - 1) # M x C x C
            # sigma = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
            # cov = cov / (sigma.unsqueeze(2) @ sigma.unsqueeze(1))
            Lambda, Q = torch.linalg.eigh(cov)
            Lambda = torch.pow(Lambda + self.EPS, -0.5)
            Std_inv = Q @ torch.diag_embed(Lambda) @ Q.mT # M x C x C

            self.mu_track = (1-self.momentum) * self.mu_track + self.momentum * x_mu.squeeze(0).clone().detach()
            self.Std_inv_track = (1-self.momentum) * self.Std_inv_track + self.momentum * Std_inv.clone().detach()
            # self.mu_track = x_mu.squeeze(0).clone().detach()
            # self.Std_inv_track = Std_inv.clone().detach()

        return Z.permute(2, 1, 0) # B x C x M

class ClusterNorm1dv4(nn.Module):

    def __init__(self, cluster_dim, num_clusters, affine=False, momentum=0.1):

        super().__init__()
        self.cluster_dim  = cluster_dim
        self.num_clusters = num_clusters
        self.affine = affine
        self.momentum = momentum
        if self.affine:
            self.scale = nn.Parameter(torch.ones(self.cluser_dim))
            self.bias  = nn.Parameter(torch.randn(self.clusre_dim)/self.num_clusters)
        self.register_buffer('mu_track', torch.randn(self.cluster_dim, self.num_clusters))
        self.register_buffer('L_track', torch.randn(self.num_clusters, self.cluster_dim, self.cluster_dim))
        self.register_buffer('EPS', torch.eye(self.cluster_dim) * 1e-4)

    def mat_shrinkage(self, C, gamma1=3, gamma2=3):
        I = torch.eye(self.cluster_dim).unsqueeze(0).to(C)
        V1 = torch.mean(torch.diagonal(C, dim1=-2, dim2=-1), dim=-1)
        V2 = torch.sum(C, dim=(-2, -1)) - torch.sum(torch.diagonal(C, dim1=-2, dim2=-1), dim=-1)
        V2 = V2 / (self.cluster_dim**2 - self.cluster_dim)
        C = C + gamma1 * V1.unsqueeze(-1).unsqueeze(-1) * I + gamma2 * V2.unsqueeze(-1).unsqueeze(-1) * (1-I)
        return C

    def forward(self, x):

        if not self.training:
            x_mu = self.mu_track.unsqueeze(0)
            x_cnt = (x - x_mu).permute(2, 1, 0)
            Z = torch.linalg.solve_triangular(self.L_track, x_cnt, upper=False)
            if self.affine:
                Z = Z * self.scale.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)
            return Z.permute(2, 1, 0)

        # TRAINING
        B = x.size(0)
        x_mu = torch.mean(x, dim=0, keepdim=True) # (B x C x M) -> (1 x C x M)
        x_cnt = x - x_mu
        x_cnt = x_cnt.permute(2, 1, 0) # (B x C x M -> M x C x B)
        cov = torch.bmm(x_cnt, x_cnt.permute(0, 2, 1)) / B # M x C x C
        cov = self.mat_shrinkage(cov)
        L = torch.linalg.cholesky(cov + self.EPS) # M x C x C
        self.mu_track = x_mu.squeeze(0).clone().detach()
        self.L_track  = L.clone().detach()
        # self.mu_track = (1-self.momentum) * self.mu_track + self.momentum * x_mu.squeeze(0).clone().detach()
        # self.L_track = (1-self.momentum) * self.L_track + self.momentum * L.clone().detach()
        Z = torch.linalg.solve_triangular(L, x_cnt, upper=False)
        if self.affine:
            Z = Z * self.scale.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)
        return Z.permute(2, 1, 0) # B x C x M


class ClusterNorm1dv5(nn.Module):

    def __init__(self, cluster_dim, num_clusters, affine=False, momentum=0.1):

        super().__init__()
        self.cluster_dim  = cluster_dim
        self.num_clusters = num_clusters
        self.affine = affine
        self.momentum = momentum
        if self.affine:
            self.scale = nn.Parameter(torch.ones(self.cluser_dim))
            self.bias  = nn.Parameter(torch.randn(self.clusre_dim)/self.num_clusters)
        self.register_buffer('mu_track', torch.randn(self.cluster_dim, self.num_clusters))
        self.register_buffer('S_inv_track', torch.randn(self.num_clusters, self.cluster_dim, self.cluster_dim))
        # self.register_buffer('EPS', torch.ones(1) * 1e-6)

    def mat_shrinkage(self, C, n):
        I = torch.eye(self.cluster_dim).unsqueeze(0).to(C)
        p = self.cluster_dim
        trace = torch.mean(torch.diagonal(C, dim1=-2, dim2=-1), dim=-1, keepdim=True).unsqueeze(-1) # Tr(C) / p
        F_hat = trace * I

        second_trace = (C * C).sum(dim=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1) # Tr(C^2) / p
        numerator = torch.pow(p * trace, 2) - second_trace # Tr^2(C) - Tr(C^2) / p
        denominator = (n - 1) * (second_trace - torch.pow(trace, 2)) # (n-1) * [Tr(C^2)/p - Tr^2(C)/p^2]
        rho_oas = torch.clamp(numerator / denominator, max=1)
        C = rho_oas * F_hat + (1-rho_oas) * C
        return C

    def forward(self, x):

        if not self.training:
            x_mu = self.mu_track.unsqueeze(0)
            x_cnt = (x - x_mu).permute(2, 1, 0)
            Z = self.S_inv_track @ x_cnt
            if self.affine:
                Z = Z * self.scale.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)
            return Z.permute(2, 1, 0)
        # TRAINING
        x = x.double()
        B = x.size(0)
        x_mu = torch.mean(x, dim=0, keepdim=True) # (B x C x M) -> (1 x C x M)
        x_cnt = x - x_mu
        x_cnt = x_cnt.permute(2, 1, 0) # (B x C x M -> M x C x B)
        cov = torch.bmm(x_cnt, x_cnt.permute(0, 2, 1)) / B # M x C x C
        cov = self.mat_shrinkage(cov, n=B)
        L, Q = torch.linalg.eigh(cov) # M x C x C
        print(L)
        L = torch.pow(L, -0.5)
        S_inv = Q @ torch.diag_embed(L) @ Q.mT # M x C x C
        Z = (S_inv @ x_cnt).float()
        self.mu_track = x_mu.squeeze(0).clone().detach().float()
        self.S_inv_track  = S_inv.clone().detach().float()
        # self.mu_track = (1-self.momentum) * self.mu_track + self.momentum * x_mu.squeeze(0).clone().detach()
        # self.L_track = (1-self.momentum) * self.L_track + self.momentum * L.clone().detach()
        if self.affine:
            Z = Z * self.scale.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)
        return Z.permute(2, 1, 0) # B x C x M


class ClusterNorm1dv6(nn.Module):

    def __init__(self, cluster_dim, num_clusters, affine=False, momentum=0.1):

        super().__init__()
        self.cluster_dim  = cluster_dim
        self.num_clusters = num_clusters
        self.affine = affine
        self.momentum = momentum
        if self.affine:
            self.scale = nn.Parameter(torch.ones(self.cluser_dim))
            self.bias  = nn.Parameter(torch.randn(self.clusre_dim)/self.num_clusters)
        self.register_buffer('mu_track', torch.randn(self.cluster_dim, self.num_clusters))
        self.register_buffer('L_track', torch.randn(self.num_clusters, self.cluster_dim, self.cluster_dim))
        self.register_buffer('EPS', torch.eye(self.cluster_dim) * 1e-4)

    def mat_shrinkage(self, C, n):
        I = torch.eye(self.cluster_dim).unsqueeze(0).to(C)
        p = self.cluster_dim
        trace = torch.mean(torch.diagonal(C, dim1=-2, dim2=-1), dim=-1, keepdim=True).unsqueeze(-1) # Tr(C) / p
        F_hat = trace * I

        second_trace = (C * C).sum(dim=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1) # Tr(C^2) / p
        numerator = torch.pow(p * trace, 2) - second_trace # Tr^2(C) - Tr(C^2) / p
        denominator = (n - 1) * (second_trace - torch.pow(trace, 2)) # (n-1) * [Tr(C^2)/p - Tr^2(C)/p^2]
        rho_oas = torch.clamp(numerator / denominator, max=1)
        C = rho_oas * F_hat + (1-rho_oas) * C
        return C

    def forward(self, x):

        if not self.training:
            x_mu = self.mu_track.unsqueeze(0)
            x_cnt = (x - x_mu).permute(2, 1, 0)
            Z = torch.linalg.solve_triangular(self.L_track, x_cnt, upper=False)
            if self.affine:
                Z = Z * self.scale.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)
            return Z.permute(2, 1, 0)

        # TRAINING
        B = x.size(0)
        x_mu = torch.mean(x, dim=0, keepdim=True) # (B x C x M) -> (1 x C x M)
        x_cnt = x - x_mu
        x_cnt = x_cnt.permute(2, 1, 0) # (B x C x M -> M x C x B)
        cov = torch.bmm(x_cnt, x_cnt.permute(0, 2, 1)) / B # M x C x C
        cov = self.mat_shrinkage(cov, n=B)
        L = torch.linalg.cholesky(cov + self.EPS) # M x C x C
        self.mu_track = x_mu.squeeze(0).clone().detach()
        self.L_track  = L.clone().detach()
        Z = torch.linalg.solve_triangular(L, x_cnt, upper=False)
        if self.affine:
            Z = Z * self.scale.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)
        return Z.permute(2, 1, 0) # B x C x M


class ClusterNormCholesky(nn.Module):

    def __init__(self, cluster_dim, affine=False):

        super().__init__()
        self.cluster_dim  = cluster_dim
        self.affine = affine
        if self.affine:
            self.scale = nn.Parameter(torch.ones(self.cluser_dim))
            self.bias  = nn.Parameter(torch.randn(self.clusre_dim)/self.num_clusters)
        # self.register_buffer('EPS', torch.eye(cluster_dim).unsqueeze(0) * 1e-6)

    def mat_shrinkage(self, C, n):
        # Rao-Blackwell Ledoit-Wolf
        I = torch.eye(self.cluster_dim).unsqueeze(0).to(C)
        p = self.cluster_dim
        trace = torch.sum(torch.diagonal(C, dim1=-2, dim2=-1), dim=-1, keepdim=True).unsqueeze(-1) # Tr(C)
        F_hat = (trace / p) * I
        second_trace = \
            torch.sum(torch.diagonal(C @ C, dim1=-2, dim2=-1), dim=-1, keepdim=True).unsqueeze(-1) # Tr(C^2)

        numerator = ((n - 2) / n) * second_trace + torch.pow(trace, 2)
        denominator = (n + 2) * (second_trace - torch.pow(trace, 2) / p)
        rho_rblw = torch.clamp(numerator / denominator, max=1)
        C = rho_rblw * F_hat + (1 - rho_rblw) * C
        return C

    def forward(self, x):

        # TRAINING
        B, C, M = x.shape
        x_mu = torch.mean(x, dim=2, keepdim=True) # (B x C x M) -> (B x C x 1)
        x_cnt = x - x_mu                          # (B x C x M)
        cov = (x_cnt @ x_cnt.mT) / M       # B x C x C
        cov = self.mat_shrinkage(cov, n=M) # B x C x C
        L = torch.linalg.cholesky(torch.linalg.inv(cov)) # B x C x C
        Z = L.mT @ x_cnt
        if self.affine:
            Z = Z * self.scale.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)
        return Z

class ClusterNormZCA(nn.Module):

    def __init__(self, cluster_dim, shrinkage=False):

        super().__init__()
        self.cluster_dim  = cluster_dim
        self.shrinkage = shrinkage
        # self.register_buffer('EPS', torch.eye(cluster_dim).unsqueeze(0) * 1e-6)

    def mat_shrinkage(self, C, n):
        # Rao-Blackwell Ledoit-Wolf
        I = torch.eye(self.cluster_dim).unsqueeze(0).to(C)
        p = self.cluster_dim
        trace = torch.sum(torch.diagonal(C, dim1=-2, dim2=-1), dim=-1, keepdim=True).unsqueeze(-1) # Tr(C)
        F_hat = (trace / p) * I
        second_trace = \
            torch.sum(torch.diagonal(C @ C, dim1=-2, dim2=-1), dim=-1, keepdim=True).unsqueeze(-1) # Tr(C^2)

        numerator = ((n - 2) / n) * second_trace + torch.pow(trace, 2)
        denominator = (n + 2) * (second_trace - torch.pow(trace, 2) / p)
        rho_rblw = torch.clamp(numerator / denominator, max=1)
        C = rho_rblw * F_hat + (1 - rho_rblw) * C
        return C

    def forward(self, x):

        # TRAINING
        B, C, M = x.shape
        x_mu = torch.mean(x, dim=2, keepdim=True) # (B x C x M) -> (B x C x 1)
        x_cnt = x - x_mu                          # (B x C x M)
        cov = (x_cnt @ x_cnt.mT) / M       # B x C x C
        if self.shrinkage:
            cov = self.mat_shrinkage(cov, n=M) # B x C x C
        L, Q = torch.linalg.eigh(cov) # B x C x C
        L = torch.clamp(L, min=1e-8)
        S_hat = Q @ torch.diag_embed(torch.rsqrt(L)) @ Q.mT
        Z = S_hat @ x_cnt # B x C x M
        return Z

