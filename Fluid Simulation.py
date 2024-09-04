import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants for field types
U_FIELD = 0
V_FIELD = 1
S_FIELD = 2


class Fluid:
    def __init__(self, density, numX, numY, h):
        self.density = density
        self.numX = numX + 2
        self.numY = numY + 2
        self.h = h
        self.u = np.zeros((self.numY, self.numX))
        self.v = np.zeros((self.numY, self.numX))
        self.newU = np.zeros((self.numY, self.numX))
        self.newV = np.zeros((self.numY, self.numX))
        self.p = np.zeros((self.numY, self.numX))
        self.div = np.zeros((self.numY, self.numX))
        self.s = np.zeros((self.numY, self.numX))
        self.epsilon = np.ones((self.numY, self.numX)) * 1e-10
        self.x = np.arange(self.numX) * h
        self.y = np.arange(self.numY) * h
        self.xcoords, self.ycoords = np.meshgrid(self.x, self.y, indexing="xy")

    def init_func(self):
        self.sx0 = Fluid.shift_right(self.s)
        self.sy0 = Fluid.shift_up(self.s)
        self.sx1 = Fluid.shift_left(self.s)
        self.sy1 = Fluid.shift_down(self.s)
        self.sum = self.sx0 + self.sx1 + self.sy1 + self.sy0

    def externalF(self, dt, Fy, Fx):
        self.v += self.s * Fy * self.sy0 * dt
        self.u += self.s * Fx * self.sx0 * dt

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

    def solve_incompressibility(self, numIters, factor):
        for _ in range(numIters):
            div = (
                self.u[1:-1, 2:]
                - self.u[1:-1, 1:-1]
                + self.v[2:, 1:-1]
                - self.v[:-2, 1:-1]
            )

            div = div * self.s[1:-1, 1:-1]
            p = -div / self.sum[1:-1, 1:-1]

            # Update velocities
            self.u[1:-1, 1:-1] -= self.sx0[1:-1, 1:-1] * p
            self.u[1:-1, 2:] += self.sx1[1:-1, 1:-1] * p
            self.v[:-2, 1:-1] -= self.sy0[1:-1, 1:-1] * p
            self.v[2:, 1:-1] += self.sy1[1:-1, 1:-1] * p
        # Update pressure
        self.p[1:-1, 1:-1] += factor * p
        self.div[1:-1, 1:-1] = div

    def extrapolate(self):
        self.v[0, :] = self.v[1, :]
        self.v[self.numY - 1, :] = self.v[self.numY - 2, :]

        self.u[:, 0] = self.u[:, 1]
        self.u[:, self.numX - 1] = self.u[:, self.numX - 2]

    def avgU(self):
        self.avgu = (
            self.u
            + Fluid.shift_left(self.u)
            + Fluid.shift_up(self.u)
            + Fluid.shift_left(Fluid.shift_up(self.u))
        ) / 4

    def avgV(self):
        self.avgv = (
            self.v
            + Fluid.shift_right(self.v)
            + Fluid.shift_down(self.v)
            + Fluid.shift_right(Fluid.shift_down(self.v))
        ) / 4

    def advectProperties(self, dt):
        self.avgU()
        self.avgV()
        self.ux = self.xcoords
        self.uy = self.ycoords
        self.vx = self.xcoords
        self.vy = self.ycoords
        self.ux = self.ux - self.u * dt
        self.uy = self.uy - self.avgv * dt
        self.vx = self.vx - self.avgu * dt
        self.vy = self.vy - self.v * dt
        np.clip(self.ux, 0, self.numX - 1, self.ux)
        np.clip(self.uy, 0, self.numY - 1, self.uy)
        np.clip(self.vx, 0, self.numX - 1, self.vx)
        np.clip(self.vy, 0, self.numY - 1, self.vy)
        RGIu = interp.RegularGridInterpolator(points=(self.y, self.x), values=self.u)
        RGIv = interp.RegularGridInterpolator(points=(self.y, self.x), values=self.v)

        pointsU = np.vstack((self.uy.ravel(), self.ux.ravel())).T
        pointsV = np.vstack((self.vy.ravel(), self.vx.ravel())).T

        # Perform interpolation
        self.NewU = RGIu(pointsU).reshape(self.numY, self.numX)
        self.NewV = RGIv(pointsV).reshape(self.numY, self.numX)
        self.u = self.u - self.u * self.s * self.sx0 + self.NewU * self.s * self.sx0
        self.v = self.v - self.v * self.s * self.sy0 + self.NewV * self.s * self.sy0

    def viscosity(self, dt, timeratio, viscosity):
        for _ in range(timeratio):
            udiff = (
                Fluid.shift_down(self.u)
                + Fluid.shift_up(self.u)
                + Fluid.shift_right(self.u)
                + Fluid.shift_left(self.u)
            ) / 4 - self.u
            vdiff = (
                Fluid.shift_down(self.v)
                + Fluid.shift_up(self.v)
                + Fluid.shift_right(self.v)
                + Fluid.shift_left(self.v)
            ) / 4 - self.v
            self.u += dt * viscosity * self.s * udiff * self.sx0 / timeratio / self.h**2
            self.v += dt * viscosity * self.s * vdiff * self.sy0 / timeratio / self.h**2

    def simulate(self, dt, Fy, Fx, numIters, factor, timeratio, viscosity):
        self.externalF(dt, Fy, Fx)
        self.p.fill(0.0)
        self.solve_incompressibility(numIters, factor)
        self.advectProperties(dt)
        self.viscosity(dt, timeratio, viscosity)
        self.extrapolate()


def create_circle_obstacle(shape, center, radius):
    y, x = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center

    distance = (x - cx) ** 2 + (y - cy) ** 2

    mask = distance <= radius**2
    return mask


def create_gravity(gravity, numY, numX):
    return np.ones((numY, numX)) * gravity


numY = 100
numX = 500
density = 1000.0
h = 1
gravity = -5.0
dt = 0.02
numIters = 200
overrelaxation = 1
viscosity = 0.0
timeratio = 5
factor = overrelaxation * density * h / dt
fluid = Fluid(density, numX, numY, h)
Fy = create_gravity(gravity, fluid.numY, fluid.numX)
Fx = np.zeros((fluid.numY, fluid.numX))


fluid.u = np.ones((fluid.numY, fluid.numX)) * 0.0
fluid.v = np.zeros((fluid.numY, fluid.numX))
fluid.s = np.ones((fluid.numY, fluid.numX))
fluid.p = np.zeros((fluid.numY, fluid.numX))
fluid.u[:, 0:2] = np.ones_like(fluid.u[:, 0:2]) * 200
center = (50, 25)
radius = 9
circle_mask = create_circle_obstacle(fluid.s.shape, center, radius)
fluid.s[circle_mask] = 1.0 * 1e-6
fluid.s[0, :] = np.ones_like(fluid.s[0, :]) * 1e-6
fluid.s[-1, :] = np.ones_like(fluid.s[-1, :]) * 1e-6
fluid.s[:, 0] = np.ones_like(fluid.s[:, 0]) * 1e-6
fluid.s[:, -1] = np.ones_like(fluid.s[:, -1]) * 1e-6

fluid.init_func()

figU, axU = plt.subplots()


U_img = axU.imshow(
    fluid.u,
    cmap="viridis",
    interpolation="nearest",
    vmin=-50,
    vmax=200,
)

figU.colorbar(U_img, ax=axU)


def update(frame):
    fluid.simulate(dt, Fy, Fx, numIters, factor, timeratio, viscosity)
    U_img.set_data(fluid.u)
    return (U_img,)


aniU = animation.FuncAnimation(
    figU, update, frames=range(100), blit=True, repeat=True, interval=20
)

# Set labels and title
axU.set_title("FLowmap")
axU.set_xlabel("X axis")
axU.set_ylabel("Y axis")


plt.show()
