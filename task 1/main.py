
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pb
from scipy.optimize import minimize
import math


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

    def find_weights(self, initial_weights):
        return gradient_descent(self.Q, initial_weights)


def gen_data1(e_value1, e_value2, disp1, disp2, number1, number2, fig):
    class1 = np.zeros(number1) - 1
    last_column1 = np.zeros(number1) + 1
    class2 = np.zeros(number2) - 1
    last_column2 = np.zeros(number2) - 1
    for i in e_value1:
        column = rand.normal(i, disp1, number1)
        class1 = np.column_stack((class1, column))
    class1 = np.column_stack((class1, last_column1))
    for i in e_value2:
        column = rand.normal(i, disp2, number2)
        class2 = np.column_stack((class2, column))
    class2 = np.column_stack((class2, last_column2))
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
        return np.vstack((class1, class2)), axes
    else:
        plt.plot(x1, y1, 'bo')
        plt.plot(x2, y2, 'ro')
        return np.vstack((class1, class2)), 0


def draw(weights, col, axes):
    def line(a, b, x):
        return a*x+b

    def plane(d, a, b, c, x, y):
        return -a*x/c-b*y/c+d/c
    if weights.shape[0] == 4:
        interval_x = np.arange (-10, 10, 1)
        interval_y = np.arange (-10, 10, 1)
        xgrid, ygrid = np.meshgrid(interval_x, interval_y)
        zgrid = plane(weights[0], weights[1], weights[2], weights[3], xgrid, ygrid)
        axes.plot_surface(xgrid, ygrid, zgrid, vmin = -10, vmax = 10, color = col, alpha=0.3)
        axes.set_zlim(-10, 10)
    if weights.shape[0] == 3:
        interval = np.arange(-10, 10, 1)
        pb.plot(interval, line(-weights[1]/weights[2], weights[0]/weights[2], interval))
        pb.xlim([-10, 10])
        pb.ylim([-10, 10])


def gradient_descent(function, initial):
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
            if (function(initial - grad * step1) > (function(initial) - eps*step*math.sqrt(np.dot(grad,grad)))):
                step1 *= delta
            else:
                status2 = 0
        return step1
    eps = 0.1
    delta = 0.5
    h = 0.0001
    step = 0.01
    number = initial.shape[0]
    grad = np.zeros(number)
    status = 1
    while status != 0:
        find_grad()
        step = choose_step(step)
        initial = initial - grad * step
        if np.all(grad*step <= 0.001) and np.all(grad*step >= -0.001):
            status = 0
    norma = map(lambda x: x*x, initial)
    norma = math.sqrt(reduce(lambda x, y: x+y, norma))
    return initial/norma


def main(Expected1, Expected2, Dispersion1, Dispersion2, Number1, Number2):
    fig = pb.figure()
    data, axes = gen_data1(Expected1, Expected2, Dispersion1, Dispersion2, Number1, Number2, fig)
    l_regression = LogisticRegression()
    l_regression.fit(data)

    weights_by_grad = np.zeros(Expected1.shape[0]+1)
    weights_by_grad = l_regression.find_weights(weights_by_grad)

    weights_by_scipy = np.zeros(Expected1.shape[0]+1)
    weights_by_scipy = minimize(l_regression.Q, weights_by_scipy, method='nelder-mead')

    norma2 = map(lambda x: x*x, weights_by_scipy.x)
    norma2 = math.sqrt(reduce(lambda x, y: x+y, norma2))
    weights_by_scipy.x /= norma2

    print weights_by_grad, weights_by_scipy.x

    draw(weights_by_grad, 'black', axes)
    draw(weights_by_scipy.x, 'yellow', axes)
    pb.show()


Expected1 = np.array([1, 4, 3])
Expected2 = np.array([4, 3, 5])
Dispersion1 = 1
Dispersion2 = 1
Number1 = 200
Number2 = 200


main(Expected1, Expected2, Dispersion1, Dispersion2, Number1, Number2)