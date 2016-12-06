import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import linprog

# distortion
sigma = 0.2

# original parameters
a_original = np.array([0.5, 0.1, 0.0])

m = 200.0
t = np.array([i*10.0/m for i in range(int(m))])


def values(a):
    # receives vector of parametres and raw of coordinates, returns raw of values
    return np.array([a[0]*np.sin(coordinate) + a[1]*coordinate + a[2] for coordinate in t])


def sum_of_array(f, X):
    # function returns sum of array elements with f implemented to them
    sum = 0
    for elem in X:
        sum += f(elem)
    return sum

y_original = values(a_original)
y_random = np.array([element + random.gauss(0, sigma) for element in y_original])
y_random[0] -= 5
y_random[-1] += 5

# minimize sum of squares. Do it via differentiation of residual function and searching for static point
# finally get system of 3 equations, it's solution is our estimation for coefficients
v1 = sum_of_array(lambda x: np.sin(x)**2, t)
v2 = sum_of_array(lambda x: x*np.sin(x), t)
v3 = sum_of_array(lambda x: np.sin(x), t)
v4 = sum_of_array(lambda x: x**2, t)
v5 = sum_of_array(lambda x: x, t)



A = np.array([[v1, v2, v3],
              [v2, v4, v5],
              [v3, v5, m]])
b = np.array([sum_of_array(lambda i: y_random[i] * np.sin(t[i]), range(int(m))),
              sum_of_array(lambda i: y_random[i] * t[i], range(int(m))),
              sum_of_array(lambda x: x, y_random)])

a_est2 = np.linalg.solve(A, b)
y_est2 = values(a_est2)
plt.plot(t, y_est2, 'm')


# minimize sum of modules using simplex algorithm
A = np.zeros([2*int(m), int(m) + 6])
c = np.ones(int(m) + 6)
b = np.zeros(2*int(m))
for i in range(6):
    c[i] = 0.0


for i in range(int(m)):
    b[2 * i] = -1*y_random[i]
    b[2 * i + 1] = y_random[i]

    A[2 * i][0] = -1*np.sin(t[i])
    A[2 * i][1] = np.sin(t[i])
    A[2 * i][2] = -1 * t[i]
    A[2 * i][3] = t[i]
    A[2 * i][4] = -1.0
    A[2 * i][5] = 1.0
    A[2 * i][i + 6] = -1.0

    A[2 * i + 1][0] = np.sin(t[i])
    A[2 * i + 1][1] = -1 * np.sin(t[i])
    A[2 * i + 1][2] = t[i]
    A[2 * i + 1][3] = -1 * t[i]
    A[2 * i + 1][4] = 1.0
    A[2 * i + 1][5] = -1.0
    A[2 * i + 1][i + 6] = -1.0


optimized = linprog(c, A, b)
print "est1:"
print optimized.message
if type(optimized.x) != float:
    a_est1 = np.array([optimized.x[0] - optimized.x[1], optimized.x[2] - optimized.x[3], optimized.x[4] - optimized.x[5]])
    print "coeff: " + str(a_est1)
    y_est1 = values(a_est1)
    plt.plot(t, y_est1, 'c')

# minimize maximum modul, using simplex
A = np.zeros([2*int(m), 7])
c = np.zeros(7)
b = np.zeros(2*int(m))
c[6] = 0


for i in range(int(m)):
    b[2 * i] = -1*y_random[i]
    b[2 * i + 1] = y_random[i]

    A[2 * i][0] = -1*np.sin(t[i])
    A[2 * i][1] = np.sin(t[i])
    A[2 * i][2] = -1 * t[i]
    A[2 * i][3] = t[i]
    A[2 * i][4] = -1.0
    A[2 * i][5] = 1.0
    A[2 * i][6] = -1.0

    A[2 * i + 1][0] = np.sin(t[i])
    A[2 * i + 1][1] = -1 * np.sin(t[i])
    A[2 * i + 1][2] = t[i]
    A[2 * i + 1][3] = -1 * t[i]
    A[2 * i + 1][4] = 1.0
    A[2 * i + 1][5] = -1.0
    A[2 * i + 1][6] = -1.0

optimized = linprog(c, A, b)
print "est3:"
print optimized.message
if type(optimized.x) != float:
    a_est3 = np.array([optimized.x[0] - optimized.x[1], optimized.x[2] - optimized.x[3], optimized.x[4] - optimized.x[5]])
    print "coeff: " + str(a_est3)
    y_est3 = values(a_est3)
    plt.plot(t, y_est3, 'b')


plt.plot(t, y_original, 'r')
plt.plot(t, y_random, 'g')
plt.show()
