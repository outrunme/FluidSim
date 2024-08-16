import numpy as np
import timeit
from itertools import product


class Field2D:
    def __init__(self, density, numX, numY, h, force):
        self.density = density
        self.numX = numX
        self.numY = numY
        self.force = force
        self.numCells = self.numX * self.numY
        self.h = h
        self.u = np.zeros((self.numY, self.numX))
        self.v = np.zeros((self.numY, self.numX))
        self.s = np.zeros((self.numY, self.numX))
        self.p = np.zeros((self.numY, self.numX))
        self.avgu = np.zeros((self.numY, self.numX))
        self.avgv = np.zeros((self.numY, self.numX))
        self.coords = np.indices((self.numY, self.numX))
        print(self.coords)

    def externalForce(self, f, dt):
        self.v += np.multiply(self.s, self.force) * dt

    @staticmethod
    def shift_right(array):
        shape = array.shape
        arrayNew = np.zeros((shape[0], shape[1] + 1))
        arrayNew[:, 1 : shape[1] + 1] = array
        arrayNew = arrayNew[:, 0 : shape[1]]
        return arrayNew

    @staticmethod
    def shift_left(array):
        shape = array.shape
        arrayNew = np.zeros((shape[0], shape[1] + 1))
        arrayNew[:, 0 : shape[1]] = array
        arrayNew = arrayNew[:, 1 : shape[1] + 1]
        return arrayNew

    @staticmethod
    def shift_down(array):
        shape = array.shape
        arrayNew = np.zeros((shape[0] + 1, shape[1]))
        arrayNew[1 : shape[0] + 1, :] = array
        arrayNew = arrayNew[0 : shape[0], :]
        return arrayNew

    @staticmethod
    def shift_up(array):
        shape = array.shape
        arrayNew = np.zeros((shape[0] + 1, shape[1]))
        arrayNew[0 : shape[0], :] = array
        arrayNew = arrayNew[1 : shape[0] + 1, :]
        return arrayNew

    def incompressibility(self, numIters):

        sx0 = Field2D.shift_right(self.s)
        sy0 = Field2D.shift_up(self.s)
        sx1 = Field2D.shift_left(self.s)
        sy1 = Field2D.shift_down(self.s)
        self.S = sx0 + sx1 + sy1 + sy0
        self.S = np.multiply(self.s, self.S)
        S_right = Field2D.shift_right(self.S)
        S_up = Field2D.shift_up(self.S)
        for i in range(numIters):
            newu = Field2D.shift_left(self.u)
            newv = Field2D.shift_down(self.v)
            self.div = newu + newv - self.u - self.v
            self.div = np.multiply(self.div, self.s)
            div_right = Field2D.shift_right(self.div)
            div_up = Field2D.shift_up(self.div)
            corr1 = np.divide(np.multiply(self.div, sx0), self.S)
            corr2 = np.divide(np.multiply(div_right, self.s), S_right)
            corr3 = np.divide(np.multiply(self.div, sy0), self.S)
            corr4 = np.divide(np.multiply(div_up, self.s), S_up)
            corr1[np.isnan(corr1)] = 0.0
            corr2[np.isnan(corr2)] = 0.0
            corr3[np.isnan(corr3)] = 0.0
            corr4[np.isnan(corr4)] = 0.0
            self.u = self.u + corr1 - corr2
            self.v = self.v + corr3 - corr4

    def avgU(
        self,
    ):
        self.avgu = (
            self.u
            + Field2D.shift_left(self.u)
            + Field2D.shift_up(self.u)
            + Field2D.shift_left(Field2D.shift_up(self.u))
        ) / 4

    def avgV(
        self,
    ):
        self.avgv = (
            self.v
            + Field2D.shift_right(self.v)
            + Field2D.shift_down(self.v)
            + Field2D.shift_right(Field2D.shift_down(self.v))
        ) / 4

    def extrapolate(
        self,
    ):
        pass


test = Field2D(1, 4, 4, 1, 1)
test.incompressibility(1)
array = [
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
    [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
    [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
    [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)],
    [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)],
]
# array = Field2D.shift_up(array)
print(array)
