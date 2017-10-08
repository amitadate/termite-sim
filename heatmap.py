from matplotlib import pyplot as PLT
from matplotlib import cm as CM
from matplotlib import mlab as ML
import numpy as NP

n = 1e5
x = y = NP.linspace(-5, 5, 100)
X, Y = NP.meshgrid(x, y)
Z1 = ML.bivariate_normal(X, Y, 2, 2, 0, 0)
Z2 = ML.bivariate_normal(X, Y, 4, 1, 1, 1)
ZD = Z2 - Z1
x = X.ravel()
y = Y.ravel()
z = ZD.ravel()
gridsize=30
PLT.subplot(111)

# if 'bins=None', then color of each hexagon corresponds directly to its count
# 'C' is optional--it maps values to x-y coordinates; if 'C' is None (default) then
# the result is a pure 2D histogram

PLT.hexbin(x, y, C=z, gridsize=gridsize, cmap=CM.jet, bins=None)
PLT.axis([x.min(), x.max(), y.min(), y.max()])

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def f(t):
    s1 = np.cos(2*np.pi*t)
    e1 = np.exp(-t)
    return np.multiply(s1,e1)


t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
t3 = np.arange(0.0, 2.0, 0.01)

fig = plt.figure(figsize=plt.figaspect(2.))

ax = fig.add_subplot(2, 1, 2, projection='3d')
X = np.arange(-20, 20, 0.25)
xlen = len(X)
Y = np.arange(-20, 20, 0.25)
ylen = len(Y)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='yellow',
        linewidth=0, antialiased=False)
scatteredpoints = ax.scatter(X[1::20, 1::20],Y[1::20, 1::20],Z[1::20, 1::20],linewidth=0, antialiased=False)

ax.set_zlim3d(-1, 1)

plt.show()

cb = PLT.colorbar()
cb.set_label('mean value')
PLT.show()

from matplotlib import style
style.use("ggplot")
from sklearn import svm

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

plt.scatter(x,y)
plt.show()

X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

y = [0,1,0,1,0,1]

clf = svm.SVC(kernel='linear', C = 1.0)

clf.fit(X,y)

print(clf.predict([0.58,0.76]))

print(clf.predict([10.58,10.76]))

w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()


#MEMORY

cdef class FooContainer:
   cdef char * data
   def __cinit__(self, char * foo_value):
       self.data = malloc(1024, sizeof(char))
       memcpy(self.data, foo_value, min(1024, len(foo_value)))

   def get(self):
       return self.data

# python part
from foo import FooContainer

f = FooContainer(Z1 = ML.bivariate_normal(X, Y, 2, 2, 0, 0),Z2 = ML.bivariate_normal(X, Y, 4, 1, 1, 1)
pid = fork()
if not pid:
   f.get() # this call will read same memory page to where
           # parent process wrote 1024 chars of self.data
           # and cython will automatically create a new python string
           # object from it and return to caller
