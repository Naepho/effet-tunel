#!/usr/bin/env python3

from constants import ctes
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def main():
    case = 5

    if case == 5:
        E = 0.5 * 1.6 * 10**(-19)

        a_list = np.arange(10**(-9), 10.5 * 10**(-9), 0.5 * 10**(-9))
        V_0_list = np.arange(0.1 * 1.6 * 10**(-19), 1.05 * 1.6 * 10**(-19), 0.05 * 10**(-19))
        print(V_0_list)

        a_len = len(a_list)
        V_0_len = len(V_0_list)
        solutions = np.zeros((a_len, V_0_len))
        for a_index in range(a_len):
            for V_0_index in range(V_0_len):
                T, t = solve_problem(1, a_list[a_index], V_0_list[V_0_index], E)
                solutions[a_index, V_0_index] = T

        for i in range(len(solutions)):
            plt.plot(V_0_list, solutions[i], '-.', label=str(np.round(a_list[i], 19)))
        plt.plot([E, E], [0, 1], 'b')
        plt.plot([V_0_list[0], V_0_list[-1]], [1, 1], 'black')
        plt.legend()
        plt.show()

    if case == 6:
        a = 2.5 * 10**(-9)
        V_0 = 0.5 * 1.6 * 10**(-19)
        E = 0.25 * 1.6 * 10**(-19)

        T1, t1 = solve_problem(1, a, V_0, E)
        T101, t101 = solve_problem(101, a, V_0, E)

        P = []
        indices = []
        for i in range(1, 101):
            P.append(Pn(i, a, V_0, E, T101, t101))
            indices.append(i)

            plt.plot(np.arange(1, 101, 1), P)
            plt.show()

            func = lambda n: Pn(n, a, V_0, E, T101, t101)

        print("")
        print("Solution : " + str(indices[np.argmin(P)]))

    """
    # Tracer les rectangles et le triangle
    # Calcul des coordonnées des points du triangle

    x_values = [ctes["x_a"], ctes["x_a"] + ctes["a"]/2, ctes["x_a"] + ctes["a"]]
    y_values = [trig_barrier(ctes["x_a"], ctes["a"], ctes["V_0"], x_values[0]),
                trig_barrier(ctes["x_a"], ctes["a"], ctes["V_0"], x_values[1]),
                trig_barrier(ctes["x_a"], ctes["a"], ctes["V_0"], x_values[2])]

    # Affichage du triangle sur le plot
    plt.plot(x_values, y_values, 'r-', label='Triangle')
    plt.xlim(ctes["x_a"] * (1 - 0.3), (ctes["x_a"] + ctes["a"]) * (1 + 0.2))

    x_vals = np.linspace(ctes["x_a"], ctes["x_a"] + ctes["a"], ctes["n"] + 1)
    for i in range(1, len(rects_h) - 1):  # Correction : utilisez la longueur de rects_h comme référence
        plt.plot([x_vals[i-1], x_vals[i], x_vals[i], x_vals[i-1], x_vals[i-1]], [0, 0, rects_h[i], rects_h[i], 0], 'b-')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Triangle et rectangles')
    plt.legend()
    plt.grid(True)
    plt.show()
    """

def Pn(n, a, V_0, E, T101, t101):
    Tn, tn = solve_problem(n, a, V_0, E)
    epn = np.abs((Tn - T101) / T101)
    Dtn = tn / t101

    return 4 * epn + Dtn

def solve_problem(n, a, V_0, E):
    start_time = timer()
    n = int(np.round(n))

    M_T = np.identity(2)
    f = lambda x: trig_barrier(ctes["x_a"], a, V_0, x)
    rects_h = np.zeros(n + 2)
    rects_l = np.zeros_like(rects_h)
    for i in range(n):
        rect_l, rect_h = rectangle(f, n, ctes["x_a"], ctes["x_a"] + a, i + 1)
        rects_h[i + 1] = rect_h
        rects_l[i + 1] = rect_l
    rects_h[0] = 0
    rects_h[-1] = 0

    x = ctes["x_a"]
    for i in range(n + 1):
        if E == rects_h[i]:
            rects_h[i] *= 1.0000001 # increment of an epsilon
        if E == rects_h[i + 1]:
            rects_h[i + 1] *= 1.0000001 # increment of an epsilon

        k1 = np.emath.sqrt(2 * ctes["m"] * (E - rects_h[i]) / ctes["hb"]**2)
        k2 = np.emath.sqrt(2 * ctes["m"] * (E - rects_h[i + 1]) / ctes["hb"]**2)

        M1 = matrix(k1, x)
        M2 = np.linalg.inv(matrix(k2, x))
        Mtot = np.matmul(M2, M1)

        M_T = np.matmul(Mtot, M_T)

        x += rects_l[i + 1]

    frac = T(M_T)

    return frac, timer() - start_time

def rectangle(f, n, x_a, x_b, r_n):
    """
    Gives width and height of rectangle based on necessary values
    f : function to approximate
    n : number of rectangles used in approximation
    x_a : start of function domain
    x_b : end of function domain
    r_n : number of desired rectangle
    """

    l = (x_b - x_a) / n
    h = f(x_a + (r_n - 1) * l + (l / 2))

    return l, h

def line(a, b, x):
    # Just the function for a line
    return a * x + b

def trig_barrier(x_a, a, V_0, x):
    slope = V_0 / (a / 2)
    if x <= x_a or x >= x_a + a:
        return 0
    if x <= x_a + a/2:
        return line(slope, 0, x - x_a)
    if x <= x_a + a:
        return line(-1 * slope, V_0, x - x_a - a/2)

def matrix(k, x):
    M = np.array([[ np.exp(1j * k * x) , np.exp(-1j * k * x)], \
                   [ 1j * k * np.exp(1j * k * x) , -1j * k * np.exp(-1j * k * x) ]])

    return M

def T(M_T):
    return np.abs(M_T[0, 0] - ((M_T[0, 1] * M_T[1, 0]) / M_T[1,1]))**2

if __name__=="__main__":
    main()
