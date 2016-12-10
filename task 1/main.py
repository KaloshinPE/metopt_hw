
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


class LogisticRegression:
    def __init__(self):
        self.number = 0

    def fit(self, train_data):
        self.row_number = train_data.shape[0]
        self.column_number = train_data.shape[1]
        self.train = np.hsplit(train_data, (0, self.column_number-1))[1]
        self.target = np.hsplit(train_data, (0, self.column_number-1))[2]

    def Q(self, w):
            s = 0
            for i in range(self.row_number):
                power = -1*self.target[i][0]*np.dot(w, self.train[i][:])
                s += math.log((1+math.exp(power)), math.e)
            return s

    def find_weights(self):
        return gradient_descent(self.Q,)


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
        x = np.zeros(number)
        x[:] = initial

        for i in range(initial.size):
            x[i] = initial[i] + h
            grad[i] = (function(x) - function(initial))/h
            x[i] = initial[i]

    def choose_step(step):
        status2 = 1
        step1 = step
        while status2==1:
            if function(initial - grad * step1) > function((initial) - eps*step*math.sqrt(np.dot(grad,grad))):
                step1 *= delta
            else:
                status2 = 0
        return step1
    eps = 0.1
    delta = 0.5
    h = 0.0001
    step = 0.01
    number = len(initial)
    grad = np.zeros(number)
    status = 1
    while status != 0:
        find_grad()
        step = choose_step(step)
        initial = initial - grad * step
        if np.all(grad*step <= 0.001) and np.all(grad*step >= -0.001):
            status = 0
    return initial


fig = pb.figure()
class1, class2 = gen_uniform_data()
axes = print_data(class1, class2, fig)
l_regression = LogisticRegression()
l_regression.fit(np.vstack((class1, class2)))

weights = l_regression.find_weights()
print weights

draw(weights, 'black', axes)
pb.show()



