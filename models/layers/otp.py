import torch
import math
# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score = 1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n-m)
 
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    return log_P - norm


def get_matching_probs_2(S, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    # S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    # S_aug[:, :m, :n] = S
    # S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    # norm = -torch.tensor(math.log(n + m), device=S.device)
    # log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    # log_a[-1] = log_a[-1] + math.log(n-m) 
    # log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_a = -torch.tensor(math.log(m), device=S.device).expand(m).contiguous().expand(batch_size, -1)
    log_b = -torch.tensor(math.log(n), device=S.device).expand(n).contiguous().expand(batch_size, -1)
    log_P = log_otp_solver(
        log_a,
        log_b,
        S, # S_aug,
        num_iters=num_iters,
        reg=reg
    )
    return log_P


