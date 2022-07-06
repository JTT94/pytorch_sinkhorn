from collections import Callable

import torch


@torch.jit.script
def squared_distances(x: torch.Tensor, y: torch.Tensor):
    D_xx = (x * x).sum(-1).unsqueeze(2)  # (B,N,1)
    D_xy = torch.matmul(x, y.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
    D_yy = (y * y).sum(-1).unsqueeze(1)  # (B,1,M)
    return D_xx - 2 * D_xy + D_yy


@torch.jit.script
def softmin(x, epsilon, log_w):
    exponent = -x / epsilon
    exponent = exponent + log_w
    return -epsilon * torch.logsumexp(exponent, 2, True)


@torch.jit.script
def sinkhorn_mapping(cost_matrix, f, log_w, epsilon):
    B, N, _ = cost_matrix.shape
    f = f.reshape(B, 1, N)
    log_w = log_w.reshape(B, 1, N)
    x = cost_matrix - f
    return softmin(x, epsilon, log_w).reshape(B, 1, N)


def sinkhorn_potentials(x: torch.Tensor,
                        logw_x: torch.Tensor,
                        y: torch.Tensor,
                        logw_y: torch.Tensor,
                        epsilon,
                        num_iterations: int = 1,
                        threshold: float = 10 ** -3,
                        cost_fn: Callable = squared_distances,
                        stable: bool = True):
    B, N = logw_x.shape
    # cost matrices
    cost_xy = cost_fn(x, y.detach())
    cost_yx = cost_fn(y, x.detach())

    if stable:
        torch.autograd.set_grad_enabled(False)

    # init potentials
    f: torch.Tensor = torch.zeros((B, 1, N))
    f = f.type_as(x)
    g: torch.Tensor = torch.zeros((B, 1, N))
    g = g.type_as(x)
    # g = sinkhorn_mapping(cost_yx, init_f, logw_x, epsilon)
    # f = sinkhorn_mapping(cost_xy, init_g, logw_y, epsilon)

    keep_going = True
    iteration = 0.
    while keep_going:
        # active_epsilon = torch.max(epsilon, active_epsilon)
        active_epsilon = epsilon
        g_: torch.Tensor = sinkhorn_mapping(cost_yx, f, logw_x, active_epsilon)
        f_: torch.Tensor = sinkhorn_mapping(cost_xy, g, logw_y, active_epsilon)
        if stable:
            f = 0.5 * (f + f_)
            g = 0.5 * (g + g_)

        f_diff: torch.Tensor = torch.norm(f_ - f)
        g_diff: torch.Tensor = torch.norm(g_ - g)
        diff: torch.Tensor = torch.max(f_diff, g_diff)

        if not stable:
            g = g_
            f = f_

        iteration: int = iteration + 1
        keep_going: bool = (iteration < num_iterations) and (diff > threshold)

    if stable:
        torch.autograd.set_grad_enabled(True)

    g = sinkhorn_mapping(cost_yx, f.detach(), logw_x, epsilon)
    f = sinkhorn_mapping(cost_xy, g.detach(), logw_y, epsilon)

    return f, g


@torch.jit.script
def transport_from_potentials(f, g, logw_x, logw_y, epsilon, cost_matrix):
    B, N = logw_x.shape
    p_matrix = f.reshape(B, N, 1) + g.reshape(B, 1, N)
    p_matrix = p_matrix - cost_matrix
    p_matrix = p_matrix / epsilon
    p_matrix = p_matrix + logw_x.reshape(B, N, 1) + logw_y.reshape(B, 1, N)
    p_matrix = p_matrix.exp()
    return p_matrix


@torch.jit.script
def transform_matrix_from_potentials(f, g, logw_x, logw_y, epsilon, cost_matrix):
    B, N = logw_x.shape
    p_matrix = f.reshape(B, N, 1) + g.reshape(B, 1, N)
    p_matrix = p_matrix - cost_matrix
    p_matrix = p_matrix / epsilon

    p_matrix = p_matrix + logw_x.reshape(B, N, 1) + logw_y.reshape(B, 1, N)
    totals = torch.logsumexp(p_matrix, 2, True)
    p_matrix = p_matrix - totals
    p_matrix = p_matrix.exp()
    return p_matrix
