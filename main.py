# -*- coding: utf-8 -*-
import sys
from pyqtgraph.Qt import QtCore, QtGui
from visualizer import NMF4DVisualizer
from nmf import *
from generator import *


def main():
    # params
    # =========================
    N = 10
    M = 50
    K = 3
    T = 10
    lr_pgd1 = 0.02
    lr_pgd2 = 0.15
    nmf_iter = 100
    nmf_clk = 50
    noise_coeff = 0
    seed = 5

    sampler = runif_matrix_normalized   # X ~ U(0, 1)
    # sampler = prnorm_matrix_normalized  # X' ~ N(0, a^2), X = max(X', eps)

    torch.manual_seed(seed)

    # models
    # =========================
    initializer = MUNMF(K=K, T=0)
    MU = MUNMF(K=K, T=1)
    PGD1 = PGDNMF(K=K, T=1, eta=lr_pgd1, order=1)
    PGD2 = PGDNMF(K=K, T=1, eta=lr_pgd2, order=2)
    initializer.eval()
    MU.eval()
    PGD1.eval()
    PGD2.eval()

    v = NMF4DVisualizer()

    # draw the ground truth matrix
    # =========================
    X, D_gt, C_gt = sampler(N, M, K, noise_coeff=noise_coeff)
    v.connect_vertices(D_gt, width=5, color=(1, 0.2, 1, 1))

    # add gaussian noise to X
    v.draw_X(X)

    # initialize
    Dinit, Cinit = initializer(X)

    # plot D0
    v.connect_vertices(Dinit, color=(.8, .8, .8, 1))

    # visualize estimations of each model
    # =========================
    def make_prediction_data(update_D, update_C, iter=50):
        Ds = []
        Dpred, Cpred = Dinit, Cinit
        for i in range(iter):
            Dpred = update_D(X=X, D=Dpred, C=Cpred)
            Dpred = normalize_col(Dpred)
            Cpred = update_C(X=X, D=Dpred, C=Cpred)
            Cpred = normalize_col(Cpred)
            Ds.append(Dpred.detach().numpy())

        return np.array(Ds)

    # MU
    # ----------
    D_MU = make_prediction_data(MU.update_D, MU.update_C, nmf_iter)
    v.draw_model_history(D_MU, rgb=(.8, .4, .2), clock=nmf_clk)

    # 1st-order PGD
    # ----------
    D_PGD1 = make_prediction_data(PGD1.update_D, PGD1.update_C, nmf_iter)
    v.draw_model_history(D_PGD1, rgb=(.5, .9, .8), clock=nmf_clk)

    # 2nd-order PGD
    # ----------
    D_PGD2 = make_prediction_data(PGD2.update_D, PGD2.update_C, nmf_iter)
    v.draw_model_history(D_PGD2, rgb=(.5, .8, .3), clock=nmf_clk)

    # random
    # ----------
    # Ds = []
    # for i in range(30):
    #     D = np.random.random((4, K))
    #     D = D / np.sum(D, axis=0)
    #     Ds.append(D)
    # Ds = np.array(Ds)
    # v.draw_model_history(Ds, rgb=(.4, .8, .2), clock=50)

    # Start Qt event loop unless running in interactive mode.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    main()
