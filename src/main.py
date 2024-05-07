#!/usr/bin/env python3

from constants import ctes
import numpy as np

def main():
    M_T = np.identity(2)
    f = lambda x: trig_barrier(ctes["x_a"], ctes["a"], ctes["V_0"], x)
    rects_h = np.zeros(ctes["n"] + 2)
    rects_l = np.zeros_like(rects_h)
    for i in range(ctes["n"]):
        rect_l, rect_h = rectangle(f, ctes["n"], ctes["x_a"], ctes["x_a"] + ctes["a"], i + 1)
        rects_h[i + 1] = rect_h
        rects_l[i + 1] = rect_l
    rects_h[0] = 0
    rects_h[-1] = 0

    print(rects_h)
    print(rects_l)

    x = ctes["x_a"]
    for i in range(ctes["n"] + 1):
        k1 = np.emath.sqrt(2 * ctes["m"] * (ctes["E"] - rects_h[i]) / ctes["hb"])
        k2 = np.emath.sqrt(2 * ctes["m"] * (ctes["E"] - rects_h[i + 1]) / ctes["hb"])

        M1 = matrix(k1, x)
        M2 = np.linalg.inv(matrix(k2, x))
        Mtot = np.matmul(M2, M1)

        M_T = np.matmul(Mtot, M_T)

        x += rects_l[i + 1]

    print(M_T)
    TE = T(M_T)
    print(TE)

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
    return M_T[0, 0] - ((M_T[0, 1] * M_T[1, 0]) / M_T[1,1])

if __name__=="__main__":
    main()
