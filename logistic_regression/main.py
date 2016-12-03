import numpy as np
from matplotlib import pyplot as plt
import random

Number_of_points = 1000
points = []

first_average_x = -10
first_dispersion_x = 7
first_average_y = -10
first_dispersion_y = 7

second_average_x = 10
second_dispersion_x = 8
second_average_y = 10
second_dispersion_y = 8

random.seed()
for i in range(Number_of_points):
    point = []
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

# flat
w = [(first_average_x + second_average_x)/2.0*(second_average_x - first_average_x) +
     (first_average_y + second_average_y)/2.0 *(second_average_y - first_average_y),
     (second_average_x - first_average_x), (second_average_y - first_average_y)]
X = np.linspace(first_average_x - 0.5*first_dispersion_x, second_average_x + 0.5*second_dispersion_x, 2)
Y = (w[0] - X*w[1])/float(w[2])
lines = plt.plot(X, Y)
plt.setp(lines, color = 'g', linewidth = 3.0)

plt.show()