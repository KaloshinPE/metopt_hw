import numpy as np
from matplotlib import pyplot as plt
import random

Number_of_points = 1000
acc = 0.005 # accurasy


first_average_x = 0.0
first_dispersion_x = 2.0
first_average_y = 10.0
first_dispersion_y = 5.0

second_average_x = 0.0
second_dispersion_x = 2.0
second_average_y = -10.0
second_dispersion_y = 2.0

# generating data randomly
points = []
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

x = [points[i][1] for i in range(len(points)) if points[i][0] == -1]
y = [points[i][2] for i in range(len(points)) if points[i][0] == -1]
plt.plot(x, y, 'bo')

x = [points[i][1] for i in range(len(points)) if points[i][0] == 1]
y = [points[i][2] for i in range(len(points)) if points[i][0] == 1]
plt.plot(x, y, 'ro')

def norm(x):
    # calculates euclidean norm of vector
    return np.sqrt(reduce(lambda a, b: a**2 + b**2, x))


def M(X, w, Xo):
    # calculates yi*<xi,w>
    return X[0]*(np.dot(np.array(X[1:]), w) - np.dot(w, Xo))


def grad_M(X, w, Xo):
    # calculates gradient fo M
    # returns two vectors - dMi/dXo and dMi/dw in X
    return -1*X[0]*w, np.array(X[0]*(np.array(X[1:]) - Xo))


def Q(w, Xo):
    # calculates value for residual function
    Q_value = 0.0
    for X in points:
        Q_value += np.log(1 + np.exp(-1*M(X, w, Xo)))
    return Q_value

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


def make_step(Xo, w, grad_Xo, grad_w):
    # calculates step via optimizing one-dimensional function F(t) = Q(Xo|w + t*gradQ(Xo, w))
    # returns new Xo, w
    min_step = acc/np.sqrt(norm(grad_Xo)**2 + norm(grad_w)**2)
    t1 = 0
    t2 = min_step
    Q1, Q2 = Q(w, Xo), 0
    while(True):
        Q2 = Q(w - t2*grad_w, Xo - t2*grad_Xo)
        if (Q2 < Q1):
            Q1 = Q2
            t1 = t2
            t2 *= 2
        else:
             break
    while True:
        if t2-t1 <= min_step:
            break
        else:
            t = (t1 + t2)/2.0
            Q_mid = Q(w - t*grad_w, Xo - t*grad_Xo)
            if(Q_mid < Q1):
                t1 = t
                Q1 = Q_mid
            else:
                t2 = t
                Q2 = Q_mid
    if t1*np.sqrt(norm(grad_Xo)**2 + norm(grad_w)**2) < acc:
        return Xo, w
    else:
        return Xo - t1*grad_Xo, w - t1*grad_w


def draw_line(Xo, w, color):
    # drawing dividing line by start point and normal vector

    scale1 = 3
    X_left = first_average_x - scale1*first_dispersion_x
    X_right = second_average_x + scale1*second_dispersion_x
    Y_top = first_average_y + scale1*first_dispersion_y
    Y_bottom = second_average_y - scale1*second_dispersion_y

    if w[1] == 0:
        direction = np.array([0, 1])
    else:
        direction = np.array([1, -1*w[0]/w[1]])
        direction = direction/norm(direction)
    if Xo[0] < X_left and direction[0] != 0:
        Xo += direction*((X_left+X_right)/2 - Xo[0])/direction[0]
    if Xo[0] > X_right and direction[0] != 0:
        Xo -= direction*(Xo[0] - (X_left+X_right)/2)/direction[0]

    scale = 15
    point1 = Xo + scale*direction
    point2 = Xo - scale * direction
    X = np.array([point1[0], point2[0]])
    Y = np.array([point1[1], point2[1]])
    lines = plt.plot(X, Y)
    plt.setp(lines, color=color, linewidth=3.0)


# initialisation of start hyperplane
# Xo - point in the hyperplane, W - normal vector to hyperplane
# Xo = np.array([(first_average_x + second_average_x)/2.0, (first_average_y + second_average_y)/2.0])
# w =  np.array([(second_average_x - first_average_x), (second_average_y - first_average_y)])
Xo = np.array([10.0, -5.0])
w = np.array([1.0, 0.0])
draw_line(Xo, w, 'g')

# gradient decent
while(True):
    grad_Xo, grad_w = grad_Q(w, Xo)
    X1, w1 = make_step(Xo, w, grad_Xo, grad_w)
    print Xo, w, np.sqrt(norm(X1-Xo)**2+ norm(w1 - w)**2)
    if np.sqrt(norm(X1-Xo)**2+ norm(w1 - w)**2) <= acc:
        break
    else:
        Xo, w = X1, w1

# plotting final hyperplane
print Xo, w
draw_line(Xo, w, 'm')

plt.show()