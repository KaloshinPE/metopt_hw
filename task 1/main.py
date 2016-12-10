
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pb

Center1 = np.array([-4, 4, 4])
Center2 = np.array([4, -3, -4])
Dispersion1 = 6
Dispersion2 = 7
Number1 = 300
Number2 = 300


def gen_normal_data():
    class1, class2 = list(), list()
    for i in range(Number1):
        buffer = [-1,]
        for j in range(len(Center1)):
            buffer.append(np.random.normal(Center1[j], Dispersion1))
        buffer.append(-1)
        class1.append(buffer)
    for i in range(Number2):
        buffer = [-1,]
        for j in range(len(Center2)):
            buffer.append(np.random.normal(Center2[j], Dispersion2))
        buffer.append(1)
        class2.append(buffer)
    return np.array(class1), np.array(class2)


def gen_uniform_data():
    Center = (Center1 + Center2)/2
    class1, class2 = list(), list()
    for i in range(Number1):
        buffer = [-1,]
        for j in range(len(Center)):
            buffer.append(Center[j] + (0.5 - np.random.sample()) * 3 * Dispersion1)
        buffer.append(-1)
        class1.append(buffer)
    for i in range(Number2):
        buffer = [-1,]
        for j in range(len(Center)):
            buffer.append(Center[j] + (0.5 - np.random.sample()) * 3 * Dispersion2)
        buffer.append(1)
        class2.append(buffer)
    return np.array(class1), np.array(class2)


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


def draw_plane(weights, color, axes):
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


def gradient_descent():
    def Q(w):
        sum = 0
        points = np.hsplit(data, (0, data.shape[1] - 1))[1]
        class_values = np.hsplit(data, (0, data.shape[1] - 1))[2]
        for i in range(data.shape[0]):
            buf = -1 * class_values[i][0] * np.dot(w, points[i])
            sum += np.log(1 + np.exp(buf))
        return sum

    def gradQ():
        grad = np.zeros(len(w))
        points = np.hsplit(data, (0, data.shape[1] - 1))[1]
        class_values = np.hsplit(data, (0, data.shape[1] - 1))[2]
        for i in range(len(w)):
            M = 0
            for j in range(data.shape[0]):
                buf = -1 * class_values[j][0] * np.dot(w, points[j])
                M += np.exp(buf) / (1 + np.exp(buf)) * -1 * class_values[j][0] * points[j][i]
            grad[i] = M
        return grad

    def choose_step(step):
        eps = 0.1
        '''step1 = step
        k = 0
        while True:
            k += 1
            print "step choise " + str(k)
            if Q(w - grad * step1) > Q((w) - eps*step*np.sqrt(np.dot(grad,grad))) and k < 10:
                step1 /= 2
            else:
                break'''
        return step

    w = np.zeros(len(Center1) + 1)
    step = 0.01
    grad = gradQ()
    itterations = 0
    while True:
        itterations += 1
        grad = gradQ()
        step = choose_step(step)
        w = w - grad * step
        #print "iteration: " + str(itterations)
        if np.linalg.norm(grad*step) < accuracy:
            break
    return w, itterations


accuracy_raw = [1.0/2/i for i in range(1000)[1:]]


x_values, y_values = list(), list()
for i in range(len(accuracy_raw)):
    print i
    accuracy = accuracy_raw[i]
    class1, class2 = gen_uniform_data()
    data = np.vstack((class1, class2))
    weights, itterations = gradient_descent()
    x_values.append(np.log(1.0/accuracy_raw[i]))
    y_values.append(itterations)
plt.clf()
plt.plot(x_values, y_values)
plt.show()