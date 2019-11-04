# -*- coding: utf-8 -*-
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import itertools
from scipy.spatial import ConvexHull
from nmf import normalize_col


class NMF4DVisualizer:
    '''Visualizes NMF as a set of 4D column vectors (3-simplex).
    Interactive 3D visualization using pyqtgraph
    '''
    def __init__(self):
        # Create a GL View widget to display data
        self.app = QtGui.QApplication([])
        self.vw = gl.GLViewWidget()
        self.vw.show()
        self.vw.setWindowTitle('NMF visualization')
        self.vw.setCameraPosition(distance=2)

        xs = [0, 0, np.sqrt(3) / 2, 1 / (2 * np.sqrt(3))]
        ys = [0, 1, 1 / 2, 1 / 2]
        zs = [0, 0, 0, np.sqrt(2 / 3)]
        self.coordinates = np.array([xs, ys, zs])

        self._draw_grid()
        self.connect_vertices(np.eye(4), convex_hull=False)

        # for plotting multiple histories
        self.model_number = 0
        self.model_counts = [0]
        self.timers = []

    def to_3d(self, data):
        '''Converts 4D array to 3D since the input is assumed to be normalized

        Args:
            data (4xN np.ndarray): original data to plot

        Returns:
            3D coordinates of the input data

        Examples:
            [1, 0, 0, 0] -> [0, 0, 0]
            [0, 1, 0, 0] -> [0, 1, 0]
            [0, 0, 1, 0] -> [sqrt(3)/2, 1/2, 0]
            [0, 0, 0, 1] -> [1/ (2*sqrt(3)), 1/2, sqrt(2/3)]
        '''
        return np.dot(self.coordinates, data).T

    def _draw_grid(self):
        '''Add a grid to the view'''
        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()

        xgrid.rotate(90, 0, 1, 0)
        ygrid.rotate(90, 1, 0, 0)
        grid_scale = 0.1
        xgrid.scale(grid_scale, grid_scale, grid_scale)
        ygrid.scale(grid_scale, grid_scale, grid_scale)
        zgrid.scale(grid_scale, grid_scale, grid_scale)
        xgrid.translate(0, 1, 1)
        ygrid.translate(1, 0, 1)
        zgrid.translate(1, 1, 0)

        self.vw.addItem(xgrid)
        self.vw.addItem(ygrid)
        self.vw.addItem(zgrid)

    def to_4d(self, data):
        '''Converts D dimensional array to 4D

        Args:
            data (DxN np.ndarray): original data to visualize

        Returns:
            4D array

        Examples:
            [1, 2, 3, 4, 5] -> [1, 2, 3, (4+5)/2] / (1+2+3+4.5)
            [1, 2, 3, 4, 5, 5] -> [1, 2, 3, (4+5+5)/2] / (1+2+3+7)
        '''
        N = data.shape[1]
        mat4d = np.zeros((4, N), dtype=float)
        for i in range(N):
            mat4d[:3, i] = data[:3, i]
            mat4d[3, i] = data[3:, i].mean()

        return normalize_col(mat4d)

    def connect_vertices(self, vertices, width=1, color=(1, 1, 1, 1),
                         convex_hull=True):
        '''Connects all vertices or simplices (convex hull)

        Args:
            vertices (4xN np.ndarray): vertices to connect
            width (int): The line width
            color (tuple): (R, G, B, a)
            convex_hull (bool): draw the convex hull

        Returns:
            lp (gl.GLLinePlotItem)
        '''

        if vertices.shape[0] != 4:
            vertices = self.to_4d(vertices)
        v_3d = self.to_3d(vertices)  # Nx3 matrix

        pos = []

        if convex_hull:
            hull = ConvexHull(v_3d[:, :2])
            for i, j in hull.simplices:
                pos.extend([v_3d[i], v_3d[j]])
        else:
            for i, j in itertools.combinations(range(v_3d.shape[0]), 2):
                pos.extend([v_3d[i], v_3d[j]])

        lp = gl.GLLinePlotItem(pos=np.array(pos),
                               width=width,
                               color=color,
                               antialias=True,
                               mode='lines')
        self.vw.addItem(lp)
        return lp

    def draw_X(self, X, size=0.01, color=(1, 1, 1, .9), pxMode=False):
        '''Draws X as a points

        Args:
            X (4xN np.ndarray): X
        '''
        if X.shape[0] != 4:
            X = self.to_4d(X)
        pos = self.to_3d(X)
        spi = gl.GLScatterPlotItem(pos=pos,
                                   color=color, size=size, pxMode=pxMode)
        self.vw.addItem(spi)

    def draw_model_history(self, models, rgb=(0, .6, .8), clock=500):
        '''Draws updating process of NMF as 2D planes

        Args:
            models (Nx4xK np.ndarray): The history of D, C
            rgb (tuple): (R, G, B) [0, 1]
            clock (int): QtCore.QTimer.start(clock)
        '''
        current_color = rgb + (1,)
        prev_color = rgb + (.7,)
        next_color = rgb + (.8,)
        others_color = rgb + (.2,)
        N = len(models)

        lp_models = []
        for m in models:
            lp_models.append(self.connect_vertices(
                m, width=1, color=others_color
            ))
        lp_models = np.array(lp_models)

        def update_model():
            count = self.model_counts[self.model_number]
            prev_model_id = (count - 1) % N
            model_id = count % N
            next_model_id = (count + 1) % N

            mask = np.ones(N, dtype=bool)
            mask[[prev_model_id, model_id, next_model_id]] = False
            for lp in lp_models[mask]:
                lp.setData(color=others_color, width=1)

            lp_models[prev_model_id].setData(color=prev_color, width=1)
            lp_models[model_id].setData(color=current_color, width=5)
            lp_models[next_model_id].setData(color=next_color, width=1)
            self.model_counts[self.model_number] += 1

        self.timers.append(QtCore.QTimer())
        self.timers[self.model_number].timeout.connect(update_model)
        self.timers[self.model_number].start(clock)

        self.model_number += 1
        self.model_counts.append(0)
