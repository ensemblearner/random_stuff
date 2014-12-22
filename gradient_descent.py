import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

def plot(x, y, function):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x,y)
    zs = np.array([function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
    ax.contour(X, Y, Z, zdir='z', cmap=cm.coolwarm)
    ax.contour(X, Y, Z, zdir='x', cmap=cm.coolwarm)
    ax.contour(X, Y, Z, zdir='y', cmap=cm.coolwarm)
    ax.set_xlim3d(-7, 7)
    ax.set_ylim3d(-7, 7)
    return Z

def gradient_descent(step_size, grad_x, grad_y, x_old, y_old, iters=50):
    x_trac, y_trac =[], []
    for iter in xrange(iters):
        x_new = x_old - step_size*grad_x(x_old, y_old)
        y_new = y_old - step_size*grad_y(x_old, y_old)
        x_old, y_old = x_new, y_new
    return x_new, y_new

def get_z(x,y, function):
    X, Y = np.meshgrid(x,y)
    zs = np.array([function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    return Z


X = np.arange(-6, 6, 0.2)
Y = np.arange(-6, 6, 0.2)
funct = lambda x,y: (x-0.2)**2 + y*y
#Z = get_z(X, Y, funct)
#print Z
Z = plot(X,Y, funct)
#plt.show()

grad_x = lambda x,y: 2*(x - 0.2)
grad_y = lambda x,y: 2*y
x_op, y_opt =gradient_descent(0.2, grad_x, grad_y, 3, 3 )
print x_op, y_opt
print funct(x_op, y_opt)

"""
func1 = lambda x, y: 1.125*x**2 + 0.5*x*y + 0.75*y**2 + 2*x - 2*y
func2 = lambda x, y: 0.5*(x**2 + y**2) + 50 * math.log(1 + math.exp(-0.5*y)) + \
                     50 * math.log(1 + math.exp(0.2*x))
func3 = lambda x, y: 0.1*(x**2 + y - 11)**2 + 0.1*(x+y**2 -7)**2
func4 = lambda x,y: 0.002*(1-x)**2 + 0.2*(y-x**2)**2
for func in [func1, func2, func3]:
    plot(func, X, Y)
    plt.show()
X = np.arange(-3,3,0.2)
plot(func4,X, Y)
plt.show()
"""


