import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1, 2]])
b = np.array([2])
c = np.array([3, 4])
m, n = A.shape[0], A.shape[1]
acc = 10**(-3)
e1, e2, e3, e4 = acc, acc, acc, acc
k_max = 300


def G_value(x, x_s, y, y_s, mu):
    r1, r2 = np.dot(A, x) + x_s - b, np.dot(A.transpose(), y) - y_s - c
    v1, v2 = np.array([x[i]*y_s[i] - mu for i in range(len(x))]), np.array([x_s[j]*y[j] - mu for j in range(len(y))])
    return np.dot(r1, r1) + np.dot(r2, r2) + np.dot(v1, v1) + np.dot(v2, v2)


def grad_G(x, x_s, y, y_s, mu):
    r1 = np.dot(A, x) + x_s - b,
    r2 = np.dot(A.transpose(), y) - y_s - c
    dot1, dot2 = np.dot(r1, np.ones(len(r1))), np.dot(r2, np.ones(len(r2)))
    grad_x, grad_x_s, grad_y, grad_y_s = np.zeros(len(x)), np.zeros(len(x_s)), np.zeros(len(y)), np.zeros(len(y_s))
    for i in range(len(x)):
        grad_x[i] = 2*np.dot(r1, A.transpose()[i]) + 2*(x[i]*y_s[i] - mu)*y_s[i]
        #grad_y_s[i] = -2*dot2 + 2*(x[i]*y_s[i] - mu)*x[i]
        grad_y_s[i] = -2 * r2[i] + 2 * (x[i] * y_s[i] - mu) * x[i]
    for j in range(len(y)):
        grad_y[j] = 2*np.dot(r2, A[j]) + 2*(x_s[j]*y[j] - mu)*x_s[j]
        grad_x_s[j] = 2 * r1[j] + 2 * (x_s[j] * y[j] - mu) * y[j]
        #grad_x_s[j] = 2*dot1 + 2*(x_s[j]*y[j] - mu)*y[j]
    return grad_x, grad_x_s, grad_y, grad_y_s


def calc_next_point(x, x_s, y, y_s, mu):
    grad_x, grad_x_s, grad_y, grad_y_s = grad_G(x, x_s, y, y_s, mu)
    X = np.hstack((x, x_s, y, y_s))
    gradG_X = np.hstack((grad_x, grad_x_s, grad_y, grad_y_s))
    G_X = G_value(x, x_s, y, y_s, mu)
    for i in range(len(X)):
        if gradG_X[i] > 0 and X[i] == 0:
            gradG_X[i] = 0
    alpha = G_X/np.linalg.norm(gradG_X)
    for i in range(len(X)):
        if gradG_X[i] > 0 and X[i] == 0:
            gradG_X[i] = 0
        if gradG_X[i] > 0:
            alpha = min(alpha, X[i]/(gradG_X[i]/np.linalg.norm(gradG_X)))
    X_next = X - G_X*gradG_X/np.linalg.norm(gradG_X)**2
    for i in range(len(X_next)):
        if X_next[i] < 0:
            X_next[i] = 0
    return X_next[:len(x)], X_next[len(x):len(x) + len(x_s)], X_next[len(x) + len(x_s):len(x) + len(x_s) + len(y)], X_next[len(x) + len(x_s) + len(y):len(x) + len(x_s) + len(y) + len(y_s)]


def solve_G(mu):
    x, y_s, x_s, y = 2*np.ones(n), np.ones(n), np.ones(m), np.ones(m)
    while True:
        x_next, x_s_next, y_next, y_s_next = calc_next_point(x, x_s, y, y_s, mu)
        if np.linalg.norm(x - x_next) <= e1 and np.linalg.norm(x_s - x_s_next) <= e2 and np.linalg.norm(y - y_next) <= e3 and np.linalg.norm(y_s - y_s_next) <= e4:
                break
        x, x_s, y, y_s = x_next, x_s_next, y_next, y_s_next
    return x, x_s, y, y_s


x, y_s, x_s, y = 2*np.ones(n), np.ones(n), np.ones(m), np.ones(m)
mu = 5
for i in range(k_max):
    x_next, x_s_next, y_next, y_s_next = solve_G(mu)
    if np.linalg.norm(x - x_next) <= e1 and np.linalg.norm(x_s - x_s_next) <= e2 and np.linalg.norm(y - y_next) <= e3 and np.linalg.norm(y_s - y_s_next) <= e4:
        break;
    x, x_s, y, y_s = x_next, x_s_next, y_next, y_s_next
    mu = mu/2

print x, x_s, y, y_s