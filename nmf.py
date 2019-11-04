import numpy as np
import torch
import torch.nn as nn


EPS = torch.finfo(torch.float).eps


def normalize_col(A):
    '''Normalize each column of a matrix (L1 norm)

    Args:
        A (torch.Tensor): The input matrix

    Returns:
        The normalized matrix

    '''
    return A / (A.sum(0) + EPS)


class BaseNMF(nn.Module):
    ''' BaseClass of NMF '''

    def __init__(self, K, T, normalize_D=False):
        super().__init__()

        self.K = K
        self.T = T
        self.normalize_D = normalize_D
        self.eps = EPS

    def _init_random_scaled(self, X, w, h):
        return torch.rand(w, h).mul(X.mean().div(self.K).sqrt()).to(X.device)

    def update_D(self, D, X, C):
        pass

    def update_C(self, C, X, D):
        pass

    def forward(self, X, D=None, C=None, verbose=False):
        if D is None:
            D = self._init_random_scaled(X, X.size(0), self.K)
        if C is None:
            C = self._init_random_scaled(X, self.K, X.size(1))
        for i in range(self.T):
            D = self.update_D(D, X, C)
            # column-wise normalization
            if self.normalize_D:
                D = normalize_col(D)
            C = self.update_C(C, X, D)
            if verbose and i % 10 == 0:
                reconst_err = self.reconstruction_err(X, D, C)
                print(f'#{i}: Reconst. error={reconst_err}')
        return D, C

    def reconstruction_err(self, X, D, C):
        return X.sub(D.matmul(C)).norm(p='fro')


class MUNMF(BaseNMF):
    ''' NMF by Multiplicative Update rule '''

    def __init__(self, **kwargs):
        super(MUNMF, self).__init__(**kwargs)

    def update_D(self, D, X, C):
        numerator = torch.matmul(X, C.t())
        CCt = torch.matmul(C, C.t())
        denominator = torch.matmul(D, CCt)
        denominator[denominator == 0] = self.eps
        return D.mul(numerator.div(denominator))

    def update_C(self, C, X, D):
        numerator = torch.matmul(D.t(), X)
        DtD = torch.matmul(D.t(), D)
        denominator = torch.matmul(DtD, C)
        denominator[denominator == 0] = self.eps
        return C.mul(numerator.div(denominator))


class RMUNMF(BaseNMF):
    ''' Randomized Multiplicative Update rule '''

    def __init__(self, N, M, **kwargs):
        super(RMUNMF, self).__init__(**kwargs)
        self.N = N
        self.M = M

    def update_D(self, D, X, C):
        indices = np.random.choice(X.size(1), self.M)
        X = X[:, indices]  # NxM
        C = C[:, indices]  # KxM

        numerator = torch.matmul(X, C.t())
        CCt = torch.matmul(C, C.t())
        denominator = torch.matmul(D, CCt)
        denominator[denominator == 0] = self.eps
        return D.mul(numerator.div(denominator))

    def update_C(self, C, X, D):
        indices = np.random.choice(X.size(0), self.N)
        X = X[indices, :]  # NxM
        D = D[indices, :]  # NxK

        numerator = torch.matmul(D.t(), X)
        DtD = torch.matmul(D.t(), D)
        denominator = torch.matmul(DtD, C)
        denominator[denominator == 0] = self.eps
        return C.mul(numerator.div(denominator))


class PGDNMF(BaseNMF):
    ''' 1st/quasi 2nd-order projected gradient descent NMF '''

    def __init__(self, eta, order, **kwargs):
        super(PGDNMF, self).__init__(**kwargs)
        self.eta = eta
        self.order = order
        if self.order == 1:
            self.update_D = self.update_D_1st
            self.update_C = self.update_C_1st
        else:
            self.update_D = self.update_D_q2nd
            self.update_C = self.update_C_q2nd

    def grad_D(self, D, X, C):
        return -X.sub(D.mm(C)).mm(C.t())

    def grad_C(self, C, X, D):
        return -D.t().mm(X.sub(D.mm(C)))

    def update_D_1st(self, D, X, C):
        return torch.relu(D.sub(self.eta * self.grad_D(D, X, C)))

    def update_C_1st(self, C, X, D):
        return torch.relu(C.sub(self.eta * self.grad_C(C, X, D)))

    def update_D_q2nd(self, D, X, C):
        return torch.relu((1 - self.eta) * D +
                          self.eta * X.mm(C.t()).mm(C.mm(C.t()).pinverse()))

    def update_C_q2nd(self, C, X, D):
        return torch.relu((1 - self.eta) * C +
                          self.eta *
                          X.t().mm(D).mm(D.t().mm(D).pinverse()).t())
