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


def draw_line(Xo, w, color):
    # drawing dividing line by start point and normal vector
    scale = 3
    X_left = first_average_x - scale*first_dispersion_x
    X_right = second_average_x + scale*second_dispersion_x
    Y_top = first_average_y + scale*first_dispersion_y
    Y_bottom = second_average_y - scale*second_dispersion_y
    if(w[0] != 0 and w[1] != 0):
        direction = [np.abs(w[1]/w[0]), 1.0]

        if((X_right - Xo[0])/direction[0] > (Y_top - Xo[1])/direction[1]):
            X_finish = Xo[0] + (Y_top - Xo[1])/direction[1]*direction[0]
        else: X_finish = X_right

        if ((Xo[0]-X_left) / direction[0] > (Xo[1]-Y_bottom) / direction[1]):
            X_start = Xo[0] - (Xo[1]-Y_bottom) / direction[1] * direction[0]
        else:
            X_start = X_left

    elif w[0] == 0:
        X_start = X_left
        X_finish = X_right
    elif w[1] == 0:
        X_start = X_left
        X_finish = X_right

    X = np.array([X_start, X_finish])
    if(w[1] != 0):
        Y = (np.dot(Xo, w) - X * w[0]) / w[1]
    else: Y = np.array([Y_bottom, Y_top])
    lines = plt.plot(X, Y)
    plt.setp(lines, color=color, linewidth=3.0)



# initialisation start hyperplane
# Xo - point in the hyperplane, W - normal vector to hyperplane
# Xo = np.array([(first_average_x + second_average_x)/2.0, (first_average_y + second_average_y)/2.0])
# w =  np.array([(second_average_x - first_average_x), (second_average_y - first_average_y)])
Xo = np.array([3.0, 5.0])
w = np.array([10.0, 0.0])
draw_line(Xo, w, 'g')

# gradient decent
for i in range(2000):
    delta_Xo, delta_w = grad_Q(w, Xo)
    Xo -= delta_Xo/np.linalg.norm(delta_Xo)
    w -= delta_w/np.linalg.norm(delta_w)

# plotting final hyperplane
draw_line(Xo, w, 'm')

plt.show()