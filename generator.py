import torch
from nmf import normalize_col


EPS = torch.finfo(torch.float).eps


def prnorm_matrix_normalized(N, M, K, noise_coeff=0):
    D = torch.randn(N, K).clamp(min=EPS)
    C = torch.randn(K, M).clamp(min=EPS)
    D = normalize_col(D)
    C = normalize_col(C)
    X = D.mm(C)
    X = torch.clamp(X + noise_coeff * torch.randn_like(X), min=EPS)
    return X, D, C


def runif_matrix_normalized(N, M, K, normalizer=None, noise_coeff=0):
    D = torch.rand(N, K)
    C = torch.rand(K, M)
    D = normalize_col(D)
    C = normalize_col(C)
    X = D.mm(C)
    X = torch.clamp(X + noise_coeff * torch.randn_like(X), min=EPS)
    return X, D, C
