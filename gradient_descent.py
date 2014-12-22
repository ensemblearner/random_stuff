import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
def sq_norm(x, y):
    return x*x + y*y

def backtracking(x, y, func,grad_x, grad_y, alpha, beta):
    m_x = grad_x(x, y)
    m_y = grad_y(x,y)
    t = 1
    while func(x-t*m_x, y-t*m_y)> func(x,y) - alpha*sq_norm(x,y):
        t *= beta
    return t

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

def has_converged(z_new, z_old, epsilon=0.0001):
    return True if math.fabs(z_new - z_old) < epsilon else False

def gradient_descent(func, step_size, grad_x, grad_y, x_old, y_old,epsilon, iters=50):
    x_trac, y_trac =[x_old], [y_old]
    for iter in xrange(iters):
        x_new = x_old - step_size*grad_x(x_old, y_old)
        y_new = y_old - step_size*grad_y(x_old, y_old)
        x_trac.append(x_new)
        y_trac.append(y_new)
        z_old = func(x_old, y_old)
        z_new = func(x_new, y_new)
        x_old, y_old = x_new, y_new
        if has_converged(z_new, z_old, epsilon):
            print "converged in %d iters"%(iter)
            break


    return x_trac, y_trac


X = np.arange(-6, 6, 0.2)
Y = np.arange(-6, 6, 0.2)
funct = lambda x,y: (x-0.2)**2 + y*y
#Z = get_z(X, Y, funct)
#print Z
Z = plot(X,Y, funct)
#plt.show()

grad_x = lambda x,y: 2*(x - 0.2)
grad_y = lambda x,y: 2*y
x_start, y_start = 4, 3
alpha = 0.2
beta = 0.2
epsilon = 0.0000001
t = backtracking(x_start, y_start, funct, grad_x, grad_y, alpha, beta)
print "t: ",t
x_s, y_s =gradient_descent(funct,t, grad_x, grad_y, x_start, y_start, epsilon)
x_opt = x_s[len(x_s) - 1]
y_opt = y_s[len(y_s) - 1]
print "optimum x: %s, y: %s"%(x_opt, y_opt)
print "value ", funct(x_opt, y_opt)

