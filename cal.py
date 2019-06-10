# import math
# import numpy as np
# # sig = 4
# # q = 0.01
# # delta = 0.00001
# # T = 10000

# # eps = math.sqrt(2*np.log(1/delta))/sig
# # # eps = 1.2
# # print("basic comp")
# # print("eps= " , eps , ", delta=" , delta)
# # print("comp= " , T*eps , ", delta=" , T*delta)


# # print("advanced comp")
# # print("eps= " , eps , ", delta=" , delta)
# # print("comp= " , eps*math.sqrt(T*np.log(1/delta)) , ", delta=" , T*delta)

# # print("amplification advanced comp")
# # print("eps= " , eps , ", delta=" , delta)
# # print("comp= " , 2*eps*q*math.sqrt(T*np.log(1/delta)) , ", delta=" , q*T*delta)

# # print("moment accountant")
# # print("eps= " , eps , ", delta=" , delta)
# # print("2*eps*q=" ,2*eps*q)
# # print("comp= " , 2*eps*q*math.sqrt(T) , ", delta=" , delta)
# x=2
# print(math.sqrt(5*x*x+27*x+25)-5*math.sqrt(x+1)-math.sqrt(x*x-4) - (4*x*x*x*x-x*x*x+2*x*x+24*x+24))

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

# draw cube
r = [-1, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="b")

# draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="r")

# draw a point
ax.scatter([0], [0], [0], color="g", s=100)

# draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

a = Arrow3D([0, 1], [0, 1], [0, 1], mutation_scale=20,
            lw=1, arrowstyle="-|>", color="k")
ax.add_artist(a)
plt.show()
