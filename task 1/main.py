
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pb
import math


Center1 = np.array([1, 4, 3])
Center2 = np.array([4, 3, 5])
Dispersion1 = 1
Dispersion2 = 1
Number1 = 50
Number2 = 50



def Q(w):
    data = np.vstack((class1, class2))
    sum = 0
    points = np.hsplit(data, (0, data.shape[1] - 1))[1]
    class_values = np.hsplit(data, (0, data.shape[1] - 1))[2]
    for i in range(data.shape[0]):
        buf = -1*class_values[i][0]*np.dot(w, points[i])
        sum += np.log(1 + math.exp(buf))
    return sum


def gen_normal_data():
    class1 = np.array([np.array(
        [-1, np.random.normal(Center1[0], Dispersion1), np.random.normal(Center1[1], Dispersion1),
         np.random.normal(Center1[2], Dispersion1), -1]) for i in range(Number1)])
    class2 = np.array([np.array(
        [-1, np.random.normal(Center2[0], Dispersion2), np.random.normal(Center2[1], Dispersion2),
         np.random.normal(Center2[2], Dispersion2), 1]) for i in range(Number2)])
    return class1, class2


def gen_uniform_data():
    Center = (Center1 + Center2)/2
    class1 = np.array([np.array(
        [-1, Center[0] + (0.5 - np.random.sample()) * 3 * Dispersion1,
         Center[1] + (0.5 - np.random.sample()) * 3 * Dispersion1,
         Center[2] + (0.5 - np.random.sample()) * 3 * Dispersion1, -1]) for i in range(Number1)])
    class2 = np.array([np.array(
        [-1, Center[0] + (0.5 - np.random.sample()) * 3 * Dispersion2,
         Center[1] + (0.5 - np.random.sample()) * 3 * Dispersion2,
         Center[2] + (0.5 - np.random.sample()) * 3 * Dispersion2, 1]) for i in range(Number2)])
    return class1, class2


def print_data(class1, class2, fig):
    x1 = class1[:, 1]
    y1 = class1[:, 2]
    x2 = class2[:, 1]
    y2 = class2[:, 2]
    if class1.shape[1] > 4:
        z1 = class1[:, 3]
        z2 = class2[:, 3]
        axes = Axes3D(fig)
        axes = fig.gca(projection='3d')
        axes.plot(x1, y1, z1, lw = 0, marker = '.', ms = 10, markevery = None, color = 'red')
        axes.plot(x2, y2, z2, lw = 0, marker = '.', ms = 10, markevery = None, color = 'green')
        return axes
    else:
        plt.plot(x1, y1, 'bo')
        plt.plot(x2, y2, 'ro')
        return 0


def draw(weights, color, axes):
    def line(a, b, x):
        return a*x+b

    def plane(d, a, b, c, x, y):
        return -a*x/c-b*y/c+d/c

    if weights.shape[0] == 4:
        interval_x = np.arange (-10, 10, 1)
        interval_y = np.arange (-10, 10, 1)
        xgrid, ygrid = np.meshgrid(interval_x, interval_y)
        zgrid = plane(weights[0], weights[1], weights[2], weights[3], xgrid, ygrid)
        axes.plot_surface(xgrid, ygrid, zgrid, vmin = -10, vmax = 10, color = color, alpha=0.3)
        axes.set_zlim(-10, 10)
    if weights.shape[0] == 3:
        interval = np.arange(-10, 10, 1)
        pb.plot(interval, line(-weights[1]/weights[2], weights[0]/weights[2], interval))
        pb.xlim([-10, 10])
        pb.ylim([-10, 10])


def gradient_descent(function):
    initial = np.zeros(len(Center1)+1)

    def find_grad():
        h = 0.0001
        x = np.zeros(len(initial))
        x[:] = initial

        for i in range(initial.size):
            x[i] = initial[i] + h
            grad[i] = (function(x) - function(initial))/h
            x[i] = initial[i]

    def choose_step(step):
        eps = 0.1
        step1 = step
        while True:
            if function(initial - grad * step1) > function((initial) - eps*step*math.sqrt(np.dot(grad,grad))):
                step1 /= 2
            else:
                break
        return step1


    step = 0.01
    grad = np.zeros(len(initial))
    while True:
        find_grad()
        step = choose_step(step)
        initial = initial - grad * step
        if np.all(grad*step <= 0.001) and np.all(grad*step >= -0.001):
            break
    return initial


fig = pb.figure()
#class1, class2 = gen_uniform_data()
class1, class2 = gen_normal_data()
axes = print_data(class1, class2, fig)

weights = gradient_descent(Q)
print weights

draw(weights, 'black', axes)
pb.show()



