import numpy as np
import matplotlib.pyplot as plt
import random
import decimal as dc

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


plt.plot(t, y_original, 'r')
plt.plot(t, y_est2, 'b')
plt.plot(t, y_random, 'g')
plt.show()
