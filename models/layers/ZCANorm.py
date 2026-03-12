'''
Wang, Wei, et al. "Backpropagation-friendly eigendecomposition." Advances in Neural Information Processing Systems 32 (2019).
https://github.com/cvlab-epfl/Power-Iteration-SVD
'''
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn


class power_iteration_once(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, v_k, num_iter=19):
        '''
        :param ctx: used to save meterials for backward.
        :param M: (M x C x C) cov matrix.
        :param v_k: initial guess of leading vector in (M x C x 1)
        :return: v_k1 leading vector.
        '''
        ctx.num_iter = num_iter
        ctx.save_for_backward(M, v_k)
        return v_k

    @staticmethod
    def backward(ctx, grad_output):
        M, v_k = ctx.saved_tensors
        dL_dvk = grad_output
        I = torch.eye(M.shape[-1], out=torch.empty_like(M))
        numerator = I - v_k.mm(torch.t(v_k))
        denominator = torch.norm(M.mm(v_k)).clamp(min=1.e-5)
        ak = numerator / denominator
        term1 = ak 
        q = M / denominator
        for i in range(1, ctx.num_iter + 1):
            ak = q.mm(ak)
            term1 += ak
        dL_dM = torch.mm(term1.mm(dL_dvk), v_k.t())
        return dL_dM, ak


class cluster_power_iteration_once(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, v_k, num_iter=19):
        '''
        :param ctx: used to save meterials for backward.
        :param M: (M x C x C) cov matrix.
        :param v_k: initial guess of leading vector in (M x C x 1)
        :return: v_k1 leading vector.
        '''
        ctx.num_iter = num_iter
        ctx.save_for_backward(M, v_k)
        return v_k

    @staticmethod
    def backward(ctx, grad_output):
        M, v_k = ctx.saved_tensors
        dL_dvk = grad_output
        # I = torch.eye(M.shape[-1], out=torch.empty_like(M))
        # numerator = I - v_k.mm(torch.t(v_k))
        # denominator = torch.norm(M.mm(v_k)).clamp(min=1.e-5)
        I = torch.eye(M.shape[-1], out=torch.empty_like(M)).unsqueeze(0) # 1 x C x C
        numerator = I - (v_k @ v_k.mT) # M x C x C
        denominator = torch.linalg.norm(M @ v_k, ord=2, dim=-2, keepdim=True)
        ak = numerator / denominator
        term1 = ak          # M x C x C
        q = M / denominator # M x C x C
        # denominator = torch.linalg.norm(M @ v_k, ord=2, dim=-2, keepdim=True).clamp(min=1e-5)
        # ak = torch.div(numerator, denominator)
        # term1 = ak.clone()  # M x C x C
        # q = torch.div(M, denominator) # M x C x C
        for i in range(1, ctx.num_iter + 1):
            # ak = q.mm(ak)
            ak = q @ ak
            term1 += ak
        # dL_dM = torch.mm(term1.mm(dL_dvk), v_k.t())
        dL_dM = term1 @ dL_dvk @ v_k.mT
        return dL_dM, ak # this works since we compute the initial v_k without gradient ...

class ZCANormSVDPI(nn.Module):
    def __init__(self, cluster_dim, num_clusters, eps=1e-4, momentum=0.1, affine=False):
        super(ZCANormSVDPI, self).__init__()
        self.cluster_dim = cluster_dim
        self.num_clusters = num_clusters
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.weight = Parameter(torch.Tensor(1, cluster_dim, 1))
        self.bias = Parameter(torch.Tensor(1, cluster_dim, 1))
        self.power_layer = power_iteration_once.apply
        self.register_buffer('running_mean', torch.zeros(1, cluster_dim, num_clusters))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = self.cluster_dim
        for i in range(self.num_clusters):
            self.register_buffer("running_subspace{}".format(i), torch.eye(length, length))
            for j in range(length):
                self.register_buffer('eigenvector{}-{}'.format(i, j), torch.ones(length, 1))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(input.dim()))

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
        self._check_input_dim(x)
        if self.training:
            B, C, M = x.size()
            assert B > 1
            assert M == self.num_clusters
            assert C == self.cluster_dim

            mu = x.mean(0, keepdim=True) # 1 x C x M
            x = x - mu                   # B x C x M # centered
            x = x.permute(2, 1, 0)       # M x C x B # centered
            cov = ((x @ x.mT) / B) + torch.eye(C).unsqueeze(0).to(x) * self.eps # M x C x C
            cov = self.mat_shrinkage(cov, n=B)
            length = C
            cov_chunks = torch.chunk(cov, M, dim=0)

            Z_list = []
            for i in range(M):
                counter_i = 0
                cov_i = cov_chunks[i].squeeze(0) # 1 x C x C => C x C
                # compute eigenvectors of subgroups no grad
                with torch.no_grad():
                    u, e, v = torch.linalg.svd(cov_i)
                    ratio = torch.cumsum(e, 0)/e.sum()
                    for j in range(length):
                        if ratio[j] >= (1 - self.eps) or e[j] <= self.eps:
                            # print('{}/{} eigen-vectors selected'.format(j + 1, length))
                            # print(e[0:counter_i])
                            break
                        eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                        # eigenvector_ij.data = v[:, j][..., None].data
                        eigenvector_ij.data = u[:, j][..., None].data
                        counter_i = j + 1

                # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
                subspace = torch.zeros_like(cov_i)
                for j in range(counter_i):
                    eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                    eigenvector_ij = self.power_layer(cov_i, eigenvector_ij)
                    lambda_ij = torch.mm(cov_i.mm(eigenvector_ij).t(), eigenvector_ij)/torch.mm(eigenvector_ij.t(), eigenvector_ij)
                    if lambda_ij < 0:
                        # print('eigenvalues: ', e)
                        # print("Warning message: negative PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                        break
                    diff_ratio = (lambda_ij - e[j]).abs()/e[j]
                    if diff_ratio > 0.1:
                        break
                    subspace += torch.mm(eigenvector_ij, torch.rsqrt(lambda_ij).mm(eigenvector_ij.t()))
                    cov_i = cov_i - torch.mm(cov_i, eigenvector_ij.mm(eigenvector_ij.t()))
                Z_i = subspace @ x[i] # (C x C) @ (C x B) => C x B
                Z_list.append(Z_i)
                # xgr = torch.mm(subspace, xg[i])
                # xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__('running_subspace' + str(i))
                    running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu

            # xr = torch.cat(xgr_list, dim=0)
            # xr = xr * self.weight + self.bias
            # xr = xr.view(C, N, H, W).transpose(0, 1)
            Z = torch.stack(Z_list, dim=0) # M x C x B
            if self.affine:
                Z = Z * self.weight + self.bias
            Z = Z.permute(2, 1, 0) # B x C x M
            return Z

        else:
            B, C, M = x.size()
            x = x - self.running_mean # B x C x M
            x = x.permute(2, 1, 0)    # M x C x B
            assert M == self.num_clusters
            Z_list = []
            for i in range(M):
                subspace = self.__getattr__('running_subspace' + str(i))
                z_i = torch.mm(subspace, x[i])
                Z_list.append(z_i)
            Z = torch.stack(Z_list, dim=0)
            if self.affine:
                Z = Z * self.weight + self.bias
            Z = Z.permute(2, 1, 0) # B x C x M
            return Z

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormSVDPI, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class ZCANormSVDPI_Cluster(nn.Module):
    def __init__(self, cluster_dim, num_clusters, eps=1e-4, momentum=0.1, affine=False):
        super(ZCANormSVDPI_Cluster, self).__init__()
        self.cluster_dim = cluster_dim
        self.num_clusters = num_clusters
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.weight = Parameter(torch.Tensor(1, cluster_dim, 1))
        self.bias = Parameter(torch.Tensor(1, cluster_dim, 1))
        self.power_layer = cluster_power_iteration_once.apply
        self.register_buffer('running_mean', torch.zeros(1, cluster_dim, num_clusters))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = self.cluster_dim
        self.register_buffer("running_std_matrix", torch.eye(self.cluster_dim).unsqueeze(0).repeat(self.num_clusters, 1, 1))
        for j in range(length):
            self.register_buffer('eigenvector-{}'.format(j), torch.ones(self.num_clusters, self.cluster_dim, 1))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(input.dim()))

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
        self._check_input_dim(x)
        if self.training:
            B, C, M = x.size()
            assert B > 1
            assert M == self.num_clusters
            assert C == self.cluster_dim

            mu = x.mean(0, keepdim=True) # 1 x C x M
            x = x - mu                   # B x C x M # centered
            x = x.permute(2, 1, 0)       # M x C x B # centered
            cov = ((x @ x.mT) / B) + torch.eye(C).unsqueeze(0).to(x) * self.eps # M x C x C
            cov = self.mat_shrinkage(cov, n=B)
            length = C
            # cov_chunks = torch.chunk(cov, M, dim=0)
            counter = 0
            # compute eigenvectors of subgroups no grad
            with torch.no_grad():
                U, L, Vh = torch.linalg.svd(cov)    # (M x C x C),  (M x C), (M x C x C)
                # ratio = torch.cumsum(e, 0)/e.sum()
                ratio = torch.cumsum(L, dim=1) / torch.sum(L, dim=1, keepdim=True)
                for j in range(length):
                    if (ratio[:, j] >= (1 - self.eps)).any() or (L[:, j] <= self.eps).any():
                        # print('{}/{} eigen-vectors selected'.format(j + 1, length))
                        # print(e[0:counter_i])
                        break
                    eigenvector_j = self.__getattr__('eigenvector-{}'.format(j))
                    eigenvector_j.data = U[:, j:j+1, :].mT.data # B x C x 1
                    counter = j + 1

            # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
            S_hat = torch.zeros_like(cov)
            for j in range(counter):
                eigenvector_j = self.__getattr__('eigenvector-{}'.format(j))
                eigenvector_j = self.power_layer(cov, eigenvector_j)
                # lambda_ij = torch.mm(cov_i.mm(eigenvector_ij).t(), eigenvector_ij)/torch.mm(eigenvector_ij.t(), eigenvector_ij)
                lambda_j = (eigenvector_j.mT @ cov @ eigenvector_j) / (eigenvector_j.mT @ eigenvector_j) # M x 1 x 1
                if (lambda_j < 0).any():
                    # print('eigenvalues: ', e)
                    # print("Warning message: negative PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                    break
                L_j = L[:, j].unsqueeze(-1).unsqueeze(-1)
                diff_ratio = (lambda_j - L_j).abs() / L_j
                if (diff_ratio > 0.1).any():
                    break
                # subspace += torch.mm(eigenvector_ij, torch.rsqrt(lambda_ij).mm(eigenvector_ij.t()))
                # cov_i = cov_i - torch.mm(cov_i, eigenvector_ij.mm(eigenvector_ij.t()))
                S_hat += eigenvector_j @ torch.rsqrt(lambda_j) @ eigenvector_j.mT
                cov = cov - (cov @ eigenvector_j @ eigenvector_j.mT)
            Z = S_hat @ x # (M x C x C) @ (M x C x B) => M x C x B

            with torch.no_grad():
                running_std_mat = self.__getattr__('running_std_matrix')
                running_std_mat.data = (1 - self.momentum) * running_std_mat.data + self.momentum * S_hat.data

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu

            if self.affine:
                Z = Z * self.weight + self.bias
            Z = Z.permute(2, 1, 0) # B x C x M
            return Z

        else:
            B, C, M = x.size()
            x = x - self.running_mean # B x C x M
            x = x.permute(2, 1, 0)    # M x C x B
            assert M == self.num_clusters
            Z_list = []
            S_hat = self.__getattr__('running_std_matrix')
            Z = S_hat @ x
            if self.affine:
                Z = Z * self.weight + self.bias
            Z = Z.permute(2, 1, 0) # B x C x M
            return Z

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormSVDPI_Cluster, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

        
class ZCANormSVDPIv2(nn.Module):
    def __init__(self, cluster_dim, num_clusters, eps=1e-5, affine=False):
        super(ZCANormSVDPIv2, self).__init__()
        self.cluster_dim = cluster_dim
        self.num_clusters = num_clusters
        self.eps = eps
        self.affine = affine
        self.weight = Parameter(torch.Tensor(1, cluster_dim, 1))
        self.bias = Parameter(torch.Tensor(1, cluster_dim, 1))
        self.power_layer = cluster_power_iteration_once.apply
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = self.cluster_dim
        for j in range(length):
            self.register_buffer('eigenvector-{}'.format(j), torch.ones(self.num_clusters, self.cluster_dim, 1),persistent=False)
            # if j == 127:
            #     self.register_buffer('eigenvector-{}'.format(j), torch.ones(self.num_clusters, self.cluster_dim, 1))
            # else:
            #     self.register_buffer('eigenvector-{}'.format(j), torch.ones(40, self.cluster_dim, 1))
 
    def reset_parameters(self):
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(input.dim()))

    # def mat_shrinkage(self, C, n):
    #     I = torch.eye(self.cluster_dim).unsqueeze(0).to(C)
    #     p = self.cluster_dim
    #     trace = torch.mean(torch.diagonal(C, dim1=-2, dim2=-1), dim=-1, keepdim=True).unsqueeze(-1) # Tr(C) / p
    #     F_hat = trace * I

    #     second_trace = (C * C).sum(dim=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1) # Tr(C^2) / p
    #     numerator = torch.pow(p * trace, 2) - second_trace # Tr^2(C) - Tr(C^2) / p
    #     denominator = (n - 1) * (second_trace - torch.pow(trace, 2)) # (n-1) * [Tr(C^2)/p - Tr^2(C)/p^2]
    #     rho_oas = torch.clamp(numerator / denominator, max=1)
    #     C = rho_oas * F_hat + (1-rho_oas) * C
    #     return C
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

    def get_whitenings(self, x):
        self._check_input_dim(x)
        B, C, M = x.size()
        assert B == 1
        assert M == self.num_clusters
        assert C == self.cluster_dim

        mu = x.mean(2, keepdim=True) # B x C x 1
        x = x - mu                   # B x C x M # centered
        cov = (x @ x.mT) / M         # B x C x C
        SIGMA = cov.clone().detach()
        cov = self.mat_shrinkage(cov, n=M)
        SIGMA_RBLW = cov.clone().detach()
        cov = cov + torch.eye(C).unsqueeze(0).to(x) * self.eps
        length = C
        counter = 0
        # compute eigenvectors of subgroups no grad
        with torch.no_grad():
            U, L, Vh = torch.linalg.svd(cov)    # (B x C x C),  (B x C), (B x C x C)
            ratio = torch.cumsum(L, dim=1) / torch.sum(L, dim=1, keepdim=True)
            for j in range(length):
                if (ratio[:, j] >= (1 - self.eps)).any() or (L[:, j] <= self.eps).any():
                    break
                eigenvector_j = self.__getattr__('eigenvector-{}'.format(j))
                eigenvector_j.data = U[:, :, j:j+1].data # B x C x 1
                counter = j + 1
        # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
        S_hat = torch.zeros_like(cov)
        SIGMA_ZCA = torch.zeros_like(cov)
        for j in range(counter):
            eigenvector_j = self.__getattr__('eigenvector-{}'.format(j))
            eigenvector_j = self.power_layer(cov, eigenvector_j)
            lambda_j = (eigenvector_j.mT @ cov @ eigenvector_j) / (eigenvector_j.mT @ eigenvector_j) # B x 1 x 1
            if (lambda_j < 0).any():
                break
            L_j = L[:, j].unsqueeze(-1).unsqueeze(-1)
            diff_ratio = (lambda_j - L_j).abs() / L_j
            if (diff_ratio > 0.1).any():
                break
            SIGMA_ZCA += eigenvector_j @ lambda_j @ eigenvector_j.mT
            S_hat += eigenvector_j @ torch.rsqrt(lambda_j) @ eigenvector_j.mT
            cov = cov - (cov @ eigenvector_j @ eigenvector_j.mT)
        # Cond Number Computation
        with torch.no_grad():
            SIGMA = SIGMA.squeeze(0); SIGMA_RBLW = SIGMA_RBLW.squeeze(0); SIGMA_ZCA = SIGMA_ZCA.squeeze(0)
            L, Q = torch.linalg.eigh(SIGMA); L = torch.clamp(L, min=1e-8)
            trans_BASE = Q @ torch.diag_embed(torch.rsqrt(L)) @ Q.mT
            L, Q = torch.linalg.eigh(SIGMA_RBLW); L = torch.clamp(L, min=1e-8)
            trans_RBLW  = Q @ torch.diag_embed(torch.rsqrt(L)) @ Q.mT
            trans_ZCA   = S_hat.squeeze(0)
        return SIGMA, SIGMA_RBLW, SIGMA_ZCA, trans_BASE, trans_RBLW, trans_ZCA

    def forward(self, x):
        self._check_input_dim(x)
        B, C, M = x.size()
        assert M == self.num_clusters
        assert C == self.cluster_dim
        # NOTE
        mu = x.mean(2, keepdim=True) # B x C x 1
        x = x - mu                   # B x C x M # centered
        cov = (x @ x.mT) / M         # B x C x C
        # cov = self.mat_shrinkage(cov, n=M)
        cov = self.mat_shrinkage(cov, n=M) + torch.eye(C).unsqueeze(0).to(x) * self.eps
        # cov = cov + torch.eye(C).unsqueeze(0).to(x) * self.eps
        length = C
        counter = 0
        # compute eigenvectors of subgroups no grad
        with torch.no_grad():
            U, L, Vh = torch.linalg.svd(cov)    # (B x C x C),  (B x C), (B x C x C)
            ratio = torch.cumsum(L, dim=1) / torch.sum(L, dim=1, keepdim=True)
            for j in range(length):
                if (ratio[:, j] >= (1 - self.eps)).any() or (L[:, j] <= self.eps).any():
                    # print('{}/{} eigen-vectors selected'.format(j + 1, length))
                    # print(e[0:counter_i])
                    break
                eigenvector_j = self.__getattr__('eigenvector-{}'.format(j))
                eigenvector_j.data = U[:, :, j:j+1].data # B x C x 1
                counter = j + 1

        # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
        S_hat = torch.zeros_like(cov)
        for j in range(counter):
            eigenvector_j = self.__getattr__('eigenvector-{}'.format(j))
            eigenvector_j = self.power_layer(cov, eigenvector_j)
            lambda_j = (eigenvector_j.mT @ cov @ eigenvector_j) / (eigenvector_j.mT @ eigenvector_j) # B x 1 x 1
            if (lambda_j < 0).any():
                # print('eigenvalues: ', e)
                # print("Warning message: negative PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                break
            L_j = L[:, j].unsqueeze(-1).unsqueeze(-1)
            diff_ratio = (lambda_j - L_j).abs() / L_j
            if (diff_ratio > 0.1).any():
                break
            S_hat += eigenvector_j @ torch.rsqrt(lambda_j) @ eigenvector_j.mT
            cov = cov - (cov @ eigenvector_j @ eigenvector_j.mT)
        Z = S_hat @ x # (B x C x C) @ (B x C x M) => B x C x M

        if self.affine:
            Z = Z * self.weight + self.bias
        return Z

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormSVDPIv2, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

        
class ZCANormSVDPIv3(nn.Module):
    def __init__(self, cluster_dim, num_clusters, eps=1e-5, affine=False):
        super(ZCANormSVDPIv3, self).__init__()
        self.cluster_dim = cluster_dim
        self.num_clusters = num_clusters
        self.eps = eps
        self.affine = affine
        self.weight = Parameter(torch.Tensor(1, cluster_dim, 1))
        self.bias = Parameter(torch.Tensor(1, cluster_dim, 1))
        self.power_layer = cluster_power_iteration_once.apply
        # self.create_dictionary()
        self.reset_parameters()
        # self.dict = self.state_dict()

    # def create_dictionary(self):
    #     length = self.cluster_dim
    #     for j in range(length):
    #         self.register_buffer('eigenvector-{}'.format(j), torch.ones(self.num_clusters, self.cluster_dim, 1))
 
    def reset_parameters(self):
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(input.dim()))

    # def mat_shrinkage(self, C, n):
    #     I = torch.eye(self.cluster_dim).unsqueeze(0).to(C)
    #     p = self.cluster_dim
    #     trace = torch.mean(torch.diagonal(C, dim1=-2, dim2=-1), dim=-1, keepdim=True).unsqueeze(-1) # Tr(C) / p
    #     F_hat = trace * I

    #     second_trace = (C * C).sum(dim=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1) # Tr(C^2) / p
    #     numerator = torch.pow(p * trace, 2) - second_trace # Tr^2(C) - Tr(C^2) / p
    #     denominator = (n - 1) * (second_trace - torch.pow(trace, 2)) # (n-1) * [Tr(C^2)/p - Tr^2(C)/p^2]
    #     rho_oas = torch.clamp(numerator / denominator, max=1)
    #     C = rho_oas * F_hat + (1-rho_oas) * C
    #     return C
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
        self._check_input_dim(x)
        B, C, M = x.size()
        assert M == self.num_clusters
        assert C == self.cluster_dim

        mu = x.mean(2, keepdim=True) # B x C x 1
        x = x - mu                   # B x C x M # centered
        cov = (x @ x.mT) / M         # B x C x C
        cov = self.mat_shrinkage(cov, n=M) + torch.eye(C).unsqueeze(0).to(x) * self.eps
        length = C
        counter = 0
        # compute eigenvectors of subgroups no grad
        with torch.no_grad():
            U, L, Vh = torch.linalg.svd(cov)    # (B x C x C),  (B x C), (B x C x C)
            ratio = torch.cumsum(L, dim=1) / torch.sum(L, dim=1, keepdim=True)
            for j in range(length):
                if (ratio[:, j] >= (1 - self.eps)).any() or (L[:, j] <= self.eps).any():
                    # print('{}/{} eigen-vectors selected'.format(j + 1, length))
                    # print(e[0:counter_i])
                    break
                # eigenvector_j = self.__getattr__('eigenvector-{}'.format(j))
                # eigenvector_j.data = U[:, :, j:j+1].data # B x C x 1
                counter = j + 1

        # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
        S_hat = torch.zeros_like(cov)
        for j in range(counter):
            # eigenvector_j = self.__getattr__('eigenvector-{}'.format(j))
            eigenvector_j = U[:, :, j:j+1]
            eigenvector_j = self.power_layer(cov, eigenvector_j)
            lambda_j = (eigenvector_j.mT @ cov @ eigenvector_j) / (eigenvector_j.mT @ eigenvector_j) # B x 1 x 1
            if (lambda_j < 0).any():
                # print('eigenvalues: ', e)
                # print("Warning message: negative PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                break
            L_j = L[:, j].unsqueeze(-1).unsqueeze(-1)
            diff_ratio = (lambda_j - L_j).abs() / L_j
            if (diff_ratio > 0.1).any():
                break
            S_hat += eigenvector_j @ torch.rsqrt(lambda_j) @ eigenvector_j.mT
            cov = cov - (cov @ eigenvector_j @ eigenvector_j.mT)
        Z = S_hat @ x # (B x C x C) @ (B x C x M) => B x C x M

        if self.affine:
            Z = Z * self.weight + self.bias
        return Z

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}'

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormSVDPIv3, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class PCANormSVDPI(nn.Module):
    def __init__(self, cluster_dim, num_clusters, eps=1e-5, affine=False):
        super(PCANormSVDPI, self).__init__()
        self.cluster_dim = cluster_dim
        self.num_clusters = num_clusters
        self.eps = eps
        self.affine = affine
        self.weight = Parameter(torch.Tensor(1, cluster_dim, 1))
        self.bias = Parameter(torch.Tensor(1, cluster_dim, 1))
        self.power_layer = cluster_power_iteration_once.apply
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = self.cluster_dim
        for j in range(length):
            self.register_buffer('eigenvector-{}'.format(j), torch.ones(self.num_clusters, self.cluster_dim, 1),persistent=False)
 
    def reset_parameters(self):
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(input.dim()))

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
        self._check_input_dim(x)
        B, C, M = x.size()
        assert M == self.num_clusters
        assert C == self.cluster_dim

        mu = x.mean(2, keepdim=True) # B x C x 1
        x = x - mu                   # B x C x M # centered
        cov = (x @ x.mT) / M         # B x C x C
        cov = self.mat_shrinkage(cov, n=M) + torch.eye(C).unsqueeze(0).to(x) * self.eps
        length = C
        counter = 0
        # compute eigenvectors of subgroups no grad
        with torch.no_grad():
            U, L, Vh = torch.linalg.svd(cov)    # (B x C x C),  (B x C), (B x C x C)

        # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
        S_hat = torch.zeros_like(cov)
        for j in range(counter):
            unit_j = nn.functional.one_hot(j * torch.ones(B).to(torch.long), num_classes=C).to(x).unsqueeze(-1)
            eigenvector_j = self.__getattr__('eigenvector-{}'.format(j))
            eigenvector_j = self.power_layer(cov, eigenvector_j)
            lambda_j = (eigenvector_j.mT @ cov @ eigenvector_j) / (eigenvector_j.mT @ eigenvector_j) # B x 1 x 1
            S_hat += unit_j @ torch.rsqrt(lambda_j) @ eigenvector_j.mT
            cov = cov - (cov @ eigenvector_j @ eigenvector_j.mT)
        Z = S_hat @ x # (B x C x C) @ (B x C x M) => B x C x M

        if self.affine:
            Z = Z * self.weight + self.bias
        return Z

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(PCANormSVDPI, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

        
class ZCANormSVDPI_No_Shrink(nn.Module):
    def __init__(self, cluster_dim, num_clusters, eps=1e-5, affine=False):
        super(ZCANormSVDPI_No_Shrink, self).__init__()
        self.cluster_dim = cluster_dim
        self.num_clusters = num_clusters
        self.eps = eps
        self.affine = affine
        self.weight = Parameter(torch.Tensor(1, cluster_dim, 1))
        self.bias = Parameter(torch.Tensor(1, cluster_dim, 1))
        self.power_layer = cluster_power_iteration_once.apply
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = self.cluster_dim
        for j in range(length):
            self.register_buffer('eigenvector-{}'.format(j), torch.ones(self.num_clusters, self.cluster_dim, 1),persistent=False)
 
    def reset_parameters(self):
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        B, C, M = x.size()
        assert M == self.num_clusters
        assert C == self.cluster_dim
        mu = x.mean(2, keepdim=True) # B x C x 1
        x = x - mu                   # B x C x M # centered
        cov = (x @ x.mT) / M         # B x C x C
        cov = cov + torch.eye(C).unsqueeze(0).to(x) * self.eps
        length = C
        counter = 0
        # compute eigenvectors of subgroups no grad
        with torch.no_grad():
            U, L, Vh = torch.linalg.svd(cov)    # (B x C x C),  (B x C), (B x C x C)
            ratio = torch.cumsum(L, dim=1) / torch.sum(L, dim=1, keepdim=True)
            for j in range(length):
                if (ratio[:, j] >= (1 - self.eps)).any() or (L[:, j] <= self.eps).any():
                    break
                eigenvector_j = self.__getattr__('eigenvector-{}'.format(j))
                eigenvector_j.data = U[:, :, j:j+1].data # B x C x 1
                counter = j + 1

        # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
        S_hat = torch.zeros_like(cov)
        for j in range(counter):
            eigenvector_j = self.__getattr__('eigenvector-{}'.format(j))
            eigenvector_j = self.power_layer(cov, eigenvector_j)
            lambda_j = (eigenvector_j.mT @ cov @ eigenvector_j) / (eigenvector_j.mT @ eigenvector_j) # B x 1 x 1
            if (lambda_j < 0).any():
                break
            L_j = L[:, j].unsqueeze(-1).unsqueeze(-1)
            diff_ratio = (lambda_j - L_j).abs() / L_j
            if (diff_ratio > 0.1).any():
                break
            S_hat += eigenvector_j @ torch.rsqrt(lambda_j) @ eigenvector_j.mT
            cov = cov - (cov @ eigenvector_j @ eigenvector_j.mT)
        Z = S_hat @ x # (B x C x C) @ (B x C x M) => B x C x M

        if self.affine:
            Z = Z * self.weight + self.bias
        return Z

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormSVDPI_No_Shrink, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


