import numpy as np
from matplotlib import pyplot as plt
import random

Number_of_points = 1000
points = []

first_average_x = -10.0
first_dispersion_x = 7.0
first_average_y = -10.0
first_dispersion_y = 7.0

second_average_x = 10.0
second_dispersion_x = 8.0
second_average_y = 10.0
second_dispersion_y = 8.0

random.seed()
for i in range(Number_of_points):
    point = list()
    point.append(random.randrange(-1, 2, 2))
    if point[0] == 1:
        point.append(random.gauss(first_average_x, first_dispersion_x))
        point.append(random.gauss(first_average_y, first_dispersion_y))
    else:
        point.append(random.gauss(second_average_x, second_dispersion_x))
        point.append(random.gauss(second_average_y, second_dispersion_y))
    points.append(point)

x = [points[i][1] for i in range(len(points)) if points[i][0] == 1]
y = [points[i][2] for i in range(len(points)) if points[i][0] == 1]
plt.plot(x, y, 'ro')

x = [points[i][1] for i in range(len(points)) if points[i][0] == -1]
y = [points[i][2] for i in range(len(points)) if points[i][0] == -1]
plt.plot(x, y, 'bo')


def M(X, w, Xo):
    # calculating yi*<xi,w>
    return X[0]*(np.dot(np.array(X[1:]), w) - np.dot(w, Xo))


def grad_M(X, w, Xo):
    # calculates gradient fo M
    # returns two vectors - dMi/dXo and dMi/dw in X
    return -1*X[0]*w, np.array(X[0]*(np.array(X[1:]) - Xo))


def grad_Q(w, Xo):
    # calculates gradient for residual function
    # returns two vectors - dMi/dXo and dMi/dw in X
    grad_Xo = np.zeros(2)
    grad_w = np.zeros(2)
    for X in points:
        exp_buf = np.exp(-1*M(X, w, Xo))
        grad_coeff = -1*exp_buf/(1 + exp_buf)
        delta_grad_Xo, delat_grad_w = grad_M(X, w, Xo)
        grad_Xo +=  grad_coeff*delta_grad_Xo
        grad_w += grad_coeff*delat_grad_w
    return grad_Xo, grad_w


# initialisation start hyperplane
# Xo - point in the hyperplane, W - normal vector to hyperplane
Xo = np.array([(first_average_x + second_average_x)/2.0, (first_average_y + second_average_y)/2.0])
w =  np.array([(second_average_x - first_average_x), (second_average_y - first_average_y)])

# plotting start hyperplane
X = np.linspace(first_average_x - 0.5*first_dispersion_x, second_average_x + 0.5*second_dispersion_x, 2)
Y = (np.dot(Xo, w) - X*w[0])/float(w[1])
lines = plt.plot(X, Y)
plt.setp(lines, color='g', linewidth=3.0)

# gradient decent
for i in range(100):
    delta_Xo, delta_w = grad_Q(w, Xo)
    Xo += delta_Xo
    w += delta_w

# plotting final hyperplane
Y = (np.dot(Xo, w) - X*w[0])/float(w[1])
lines = plt.plot(X, Y)
plt.setp(lines, color='black', linewidth=3.0)

plt.show()